import torch
import torchaudio
import pandas as pd
import evaluate
import os
from transformers import (
    pipeline,
    Wav2Vec2ForCTC,
    Wav2Vec2Processor,
    AutoTokenizer,
    AutoModelForCausalLM,
    WhisperForConditionalGeneration,
    WhisperProcessor
)
from tqdm import tqdm

# =====================================================================================
# ШАГ 1: КОНФИГУРАЦИЯ ПУТЕЙ И МОДЕЛЕЙ
# =====================================================================================
# --- Пути к дообученным моделям ---
WHISPER_FT_PATH = "../../models/whisper-fine-tuned-a100/"
WAV2VEC2_FT_PATH = "../../models/wav2vec2-fine-tuned-a100/"

# --- Имена базовых (исходных) моделей для сравнения ---
WHISPER_BASE_MODEL = "openai/whisper-large-v3"
WAV2VEC2_BASE_MODEL = "jonatasgrosman/wav2vec2-large-xlsr-53-russian"

# --- Модель для ансамблирования ---
YANDEX_LLM_MODEL = "yandex/YandexGPT-5-Lite-8B-instruct"

# --- Путь к данным ---
DATA_CSV_PATH = "../../data/clear_rez_transcriptions.csv"


# =====================================================================================
# ШАГ 2: КЛАССЫ-ОБЕРТКИ ДЛЯ ВСЕХ МОДЕЛЕЙ
# =====================================================================================
# Мы создадим классы для каждой модели, чтобы инкапсулировать логику
# и легко распределить их по разным GPU.

class ASRModel:
    """Базовый класс для моделей распознавания речи."""
    def init(self, model_path_or_name, device="cuda:0", model_type="whisper"):
        self.device = device
        self.model_path = model_path_or_name
        self.model_type = model_type
        self.torch_dtype = torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float16
        print(f"Инициализация {model_type} модели '{model_path_or_name}' на устройстве {device} с dtype {self.torch_dtype}...")
        
        if model_type == "whisper":
            model = WhisperForConditionalGeneration.from_pretrained(model_path_or_name, torch_dtype=self.torch_dtype).to(self.device)
            processor = WhisperProcessor.from_pretrained(model_path_or_name)
            self.pipe = pipeline(
                "automatic-speech-recognition",
                model=model, tokenizer=processor.tokenizer, feature_extractor=processor.feature_extractor,
                device=self.device,
            )
        elif model_type == "wav2vec2":
            self.model = Wav2Vec2ForCTC.from_pretrained(model_path_or_name, torch_dtype=self.torch_dtype).to(self.device)
            self.processor = Wav2Vec2Processor.from_pretrained(model_path_or_name)
        else:
            raise ValueError("Неверный тип модели. Допустимые: 'whisper' или 'wav2vec2'")
            
    def transcribe(self, audio_path: str) -> str:
        if self.model_type == "whisper":
            result = self.pipe(audio_path, generate_kwargs={"language": "russian", "task": "transcribe"})
            return result["text"].strip()
        
        elif self.model_type == "wav2vec2":
            waveform, sr = torchaudio.load(audio_path)
            if sr != 16_000:
                waveform = torchaudio.transforms.Resample(orig_freq=sr, new_freq=16_000)(waveform)
            
            inputs = self.processor(waveform.squeeze(), sampling_rate=16_000, return_tensors="pt", padding=True)
            with torch.no_grad():
                logits = self.model(inputs.input_values.to(self.device)).logits
            
            pred_ids = torch.argmax(logits, dim=-1)
            return self.processor.batch_decode(pred_ids)[0].strip().lower()

class LocalYandexInstructEnsembler:
    def init(self, model_name, device="cuda:0"):
        self.device = device
        self.torch_dtype = torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float16
        print(f"Инициализация LLM Ensembler '{model_name}' на устройстве {device} с dtype {self.torch_dtype}...")
        self.model = AutoModelForCausalLM.from_pretrained(model_name, torch_dtype=self.torch_dtype, device_map=self.device)
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.generation_kwargs = {"max_new_tokens": 1024, "temperature": 0.1}

    def _create_chat_messages(self, trans_a, trans_b):
        return [
            {"role": "system", "content": "Ты — эксперт-редактор, исправляющий расшифровки переговоров железнодорожных диспетчеров. Проанализируй две версии, объедини их сильные стороны, исправь ошибки в числах и терминах, и выдай единственную, наилучшую и наиболее точную расшифровку без дополнительных комментариев."},
            {"role": "user", "content": f'Версия А: "{trans_a}"\n\nВерсия Б: "{trans_b}"'}
        ]

    @torch.no_grad()
    def combine(self, trans_whisper, trans_wav2vec2):
        messages = self._create_chat_messages(trans_whisper, trans_wav2vec2)
        input_ids = self.tokenizer.apply_chat_template(messages, tokenize=True, add_generation_prompt=True, return_tensors="pt").to(self.device)
        outputs = self.model.generate(input_ids, **self.generation_kwargs)
        response_ids = outputs[0][input_ids.shape[1]:]
        return self.tokenizer.decode(response_ids, skip_special_tokens=True).strip()

# =====================================================================================
# ШАГ 3: ОСНОВНАЯ ЛОГИКА ОЦЕНКИ
# =====================================================================================
if name == "main":
    # Распределяем модели по двум GPU для параллельной работы
    print("--- Шаг 3.1: Инициализация всех моделей ---")
    base_whisper_model = ASRModel(WHISPER_BASE_MODEL, device="cuda:0", model_type="whisper")
    ft_whisper_model   = ASRModel(WHISPER_FT_PATH, device="cuda:0", model_type="whisper")
    
    base_wav2vec2_model = ASRModel(WAV2VEC2_BASE_MODEL, device="cuda:1", model_type="wav2vec2")
    ft_wav2vec2_model  = ASRModel(WAV2VEC2_FT_PATH, device="cuda:1", model_type="wav2vec2")
    
    yandex_ensembler = LocalYandexInstructEnsembler(YANDEX_LLM_MODEL, device="cuda:0")

    # Загружаем данные и отбираем ТОЛЬКО тестовую выборку
    print("\n--- Шаг 3.2: Загрузка тестового набора данных ---")
    df = pd.read_csv(DATA_CSV_PATH)
    test_df = df[df['test'] == 1].copy()
    if len(test_df) == 0:
        raise ValueError("В CSV не найдено записей для теста (test == 1). Оценка невозможна.")
    print(f"Найдено {len(test_df)} файлов для финальной оценки.")

    # Прогоняем все модели на тестовом наборе
    print("\n--- Шаг 3.3: Запуск оценки на тестовом наборе ---")
    results_list = []
    
    for _, row in tqdm(test_df.iterrows(), total=test_df.shape[0], desc="Оценка моделей"):
        audio_file_relative = row["Путь к файлу"]
        audio_file_full_path = f"../../data/{audio_file_relative}"
        ground_truth = row["Расшифровка"]
        
        if not os.path.exists(audio_file_full_path):
            print(f"ПРЕДУПРЕЖДЕНИЕ: Файл не найден, пропуск: {audio_file_full_path}")
            continue

        try:
            # Получаем транскрипции от всех 4 моделей
            base_whisper_res = base_whisper_model.transcribe(audio_file_full_path)
            base_wav2vec2_res = base_wav2vec2_model.transcribe(audio_file_full_path)
            ft_whisper_res = ft_whisper_model.transcribe(audio_file_full_path)
            ft_wav2vec2_res = ft_wav2vec2_model.transcribe(audio_file_full_path)
            
            # Ансамбль работает на результатах лучших, дообученных моделей
            ensemble_res = yandex_ensembler.combine(ft_whisper_res, ft_wav2vec2_res)
            
            results_list.append({
                "audio_file": audio_file_relative,
                "ground_truth": ground_truth,
                "base_whisper": base_whisper_res,

"base_wav2vec2": base_wav2vec2_res,
                "ft_whisper": ft_whisper_res,
                "ft_wav2vec2": ft_wav2vec2_res,
                "ensemble": ensemble_res
            })
        except Exception as e:
            print(f"Ошибка при обработке файла {audio_file_relative}: {e}")
            
    # Сохраняем все результаты в один CSV для анализа
    results_df = pd.DataFrame(results_list)
    results_df.to_csv("final_evaluation_results.csv", index=False, encoding='utf-8-sig')
    print("\n--- Шаг 3.4: Все результаты сохранены в final_evaluation_results.csv ---")

    # Считаем и выводим итоговые метрики
    wer_metric = evaluate.load("wer")
    
    wer_base_whisper = 100 * wer_metric.compute(predictions=results_df["base_whisper"], references=results_df["ground_truth"])
    wer_base_wav2vec2 = 100 * wer_metric.compute(predictions=results_df["base_wav2vec2"], references=results_df["ground_truth"])
    wer_ft_whisper = 100 * wer_metric.compute(predictions=results_df["ft_whisper"], references=results_df["ground_truth"])
    wer_ft_wav2vec2 = 100 * wer_metric.compute(predictions=results_df["ft_wav2vec2"], references=results_df["ground_truth"])
    wer_ensemble = 100 * wer_metric.compute(predictions=results_df["ensemble"], references=results_df["ground_truth"])

    print("\n" + "="*80)
    print("ИТОГОВЫЕ РЕЗУЛЬТАТЫ СРАВНЕНИЯ (Word Error Rate, %):")
    print(f"  Меньше - лучше\n" + "-"*80)
    print(f"ИСХОДНЫЕ МОДЕЛИ (БЕЗ ДООБУЧЕНИЯ):")
    print(f"  - Base Whisper:       {wer_base_whisper:.2f}%")
    print(f"  - Base Wav2Vec2:      {wer_base_wav2vec2:.2f}%")
    print("-" * 80)
    print(f"МОДЕЛИ ПОСЛЕ ДООБУЧЕНИЯ:")
    print(f"  - Fine-Tuned Whisper: {wer_ft_whisper:.2f}%")
    print(f"  - Fine-Tuned Wav2Vec2:{wer_ft_wav2vec2:.2f}%")
    print("-" * 80)
    print(f"ФИНАЛЬНЫЙ АНСАМБЛЬ:")
    print(f"  - YandexGPT Ensemble:   {wer_ensemble:.2f}%")
    print("="*80)