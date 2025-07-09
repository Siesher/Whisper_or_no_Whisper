import pandas as pd
import torch
import re
import json
import os
from dataclasses import dataclass
from typing import Dict, List, Union
from datasets import Dataset, DatasetDict, Audio
from transformers import (
    Wav2Vec2ForCTC,
    Wav2Vec2Processor,
    TrainingArguments,
    Trainer,
)

# =====================================================================================
# ШАГ 1: КОНФИГУРАЦИЯ ПРОЕКТА
# =====================================================================================
# Имя базовой модели с Hugging Face
MODEL_NAME = "jonatasgrosman/wav2vec2-large-xlsr-53-russian"
# Директория для сохранения дообученной модели и чекпоинтов
OUTPUT_DIR = "../../models/wav2vec2-fine-tuned-a100"
# Путь к файлу с разметкой
DATA_CSV_PATH = "../../data/clear_rez_transcriptions.csv"
# Количество ядер CPU для параллельной обработки данных
CPU_CORES = 60

# =====================================================================================
# ШАГ 2: ЗАГРУЗКА И ПОДГОТОВКА ДАННЫХ
# =====================================================================================
print("--- Шаг 2: Загрузка и подготовка данных ---")
df = pd.read_csv(DATA_CSV_PATH)

# Проверяем наличие необходимых столбцов
required_columns = ["Путь к файлу", "Расшифровка", "test"]
if not all(col in df.columns for col in required_columns):
    raise ValueError(f"CSV файл должен содержать столбцы: {required_columns}")

# Оставляем только нужные столбцы и удаляем строки с пропусками
df = df[required_columns].dropna()
df.rename(columns={"Путь к файлу": "audio", "Расшифровка": "text"}, inplace=True)

# Формируем полные пути к аудиофайлам относительно корня проекта
df['audio'] = df['audio'].apply(lambda path: f"../../data/{path}")

# Очищаем текст от пунктуации и приводим к нижнему регистру
chars_to_ignore_regex = r'[\,\?\.\!\-\;\:\"\«\»]'
df["text"] = df["text"].apply(lambda t: re.sub(chars_to_ignore_regex, '', t).lower())

# Разделяем данные на обучающую и валидационную выборки по столбцу 'test'
print("Разделение данных по столбцу 'test'...")
train_df = df[df['test'] == 0].copy()
eval_df = df[df['test'] == 1].copy()

if len(train_df) == 0 or len(eval_df) == 0:
    raise ValueError("Не удалось сформировать обучающую или валидационную выборку. Проверьте столбец 'test' в CSV.")

raw_datasets = DatasetDict({
    "train": Dataset.from_pandas(train_df),
    "eval": Dataset.from_pandas(eval_df)
})

print(f"Размер обучающей выборки: {len(raw_datasets['train'])}")
print(f"Размер валидационной выборки: {len(raw_datasets['eval'])}")

# =====================================================================================
# ШАГ 3: СОЗДАНИЕ СЛОВАРЯ И ПРОЦЕССОРА
# =====================================================================================
print("\n--- Шаг 3: Создание словаря и процессора ---")
# Создаем словарь только на основе обучающих данных, чтобы избежать утечки данных
vocab_text = " ".join(train_df["text"])
vocab_list = list(set(vocab_text))
vocab_dict = {v: k for k, v in enumerate(vocab_list)}

# Заменяем пробел на специальный токен, как это принято в Wav2Vec2
vocab_dict["|"] = vocab_dict.pop(" ")

# Добавляем служебные токены
vocab_dict["[UNK]"] = len(vocab_dict) # Неизвестный токен
vocab_dict["[PAD]"] = len(vocab_dict) # Токен для выравнивания длины

# Сохраняем словарь в JSON файл
vocab_path = "vocab_wav2vec2.json"
with open(vocab_path, "w", encoding='utf-8') as f:
    json.dump(vocab_dict, f, ensure_ascii=False)
print(f"Словарь сохранен в: {vocab_path}")

# Создаем процессор, который объединяет feature_extractor и tokenizer
processor = Wav2Vec2Processor.from_pretrained(MODEL_NAME, vocab_path=vocab_path)

# =====================================================================================
# ШАГ 4: ПРЕДОБРАБОТКА ДАТАСЕТА
# =====================================================================================
print("\n--- Шаг 4: Предобработка датасета ---")
# Приводим аудиоколонку к нужному типу перед многопоточной обработкой
raw_datasets = raw_datasets.cast_column("audio", Audio(sampling_rate=16_000))

def prepare_dataset(batch):
    # Извлекаем признаки из аудио
    batch["input_values"] = processor(batch["audio"]["array"], sampling_rate=16_000).input_values[0]
    # Кодируем текст в виде числовых меток
    with processor.as_target_processor():
        batch["labels"] = processor(batch["text"]).input_ids
    return batch

print(f"Запуск многопоточной обработки с использованием {CPU_CORES} ядер...")
tokenized_datasets = raw_datasets.map(
    prepare_dataset,
    remove_columns=raw_datasets["train"].column_names,
    num_proc=CPU_CORES
)
print("Обработка завершена.")

# =====================================================================================
# ШАГ 5: НАСТРОЙКА ОБУЧЕНИЯ (DATA COLLATOR, TRAINER)
# =====================================================================================
print("\n--- Шаг 5: Настройка компонентов для обучения ---")
@dataclass
class DataCollatorCTCWithPadding:
    processor: Wav2Vec2Processor
    padding: Union[bool, str] = True
    def call(self, features: List[Dict[str, Union[List[int], torch.Tensor]]]) -> Dict[str, torch.Tensor]:
        input_features = [{"input_values": feature["input_values"]} for feature in features]
        label_features = [{"input_ids": feature["labels"]} for feature in features]
        batch = self.processor.pad(input_features, padding=self.padding, return_tensors="pt")
        with self.processor.as_target_processor():
            labels_batch = self.processor.pad(label_features, padding=self.padding, return_tensors="pt")
        labels = labels_batch["input_ids"].masked_fill(labels_batch.attention_mask.ne(1), -100)
        batch["labels"] = labels
        return batch

data_collator = DataCollatorCTCWithPadding(processor=processor, padding=True)

print("Загрузка модели для дообучения...")
model = Wav2Vec2ForCTC.from_pretrained(
    MODEL_NAME,
    ctc_loss_reduction="mean",
    pad_token_id=processor.tokenizer.pad_token_id,
    vocab_size=len(processor.tokenizer)
)

# Замораживаем веса feature extractor, так как он уже хорошо обучен
model.freeze_feature_encoder()

# Настраиваем аргументы для обучения на 2x A100
training_args = TrainingArguments(
    output_dir=OUTPUT_DIR,
    group_by_length=True,
    per_device_train_batch_size=16,
    per_device_eval_batch_size=16,
    bf16=True,  # Используем bfloat16, оптимизированный для A100
    max_steps=1500,  # Оптимальное количество шагов для ~5-часового пайплайна
    learning_rate=1e-4,
    warmup_steps=200,
    save_steps=500,  # Сохраняем чекпоинт каждые 500 шагов
    eval_steps=500,
    logging_steps=50,
    save_total_limit=3,  # Храним 3 последних чекпоинта
    evaluation_strategy="steps",
    save_strategy="steps",
    report_to=["tensorboard"],
)

trainer = Trainer(
    model=model,
    data_collator=data_collator,
    args=training_args,
    train_dataset=tokenized_datasets["train"],
    eval_dataset=tokenized_datasets["eval"],
    tokenizer=processor.feature_extractor,
)

# =====================================================================================
# ШАГ 6: ЗАПУСК ОБУЧЕНИЯ С ПОДДЕРЖКОЙ ВОЗОБНОВЛЕНИЯ
# =====================================================================================
print("\n--- Шаг 6: Запуск обучения ---")
last_checkpoint = None
# Проверяем, существуют ли чекпоинты в директории вывода, для возобновления
if os.path.isdir(training_args.output_dir):
    checkpoints = [d for d in os.listdir(training_args.output_dir) if d.startswith("checkpoint-")]
    if checkpoints:
        # Находим самый последний чекпоинт по номеру шага
        last_checkpoint_name = max(checkpoints, key=lambda x: int(x.split('-')[-1]))
        last_checkpoint = os.path.join(training_args.output_dir, last_checkpoint_name)
        print(f"Найден чекпоинт: {last_checkpoint}. Обучение будет возобновлено с него.")

# trainer.train() автоматически подхватит последний чекпоинт.
# Передача параметра resume_from_checkpoint делает это поведение явным.
trainer.train(resume_from_checkpoint=last_checkpoint)

print("\n--- Обучение успешно завершено. Сохранение финальной модели... ---\n")