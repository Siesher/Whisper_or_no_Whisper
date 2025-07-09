import pandas as pd
import torch
import evaluate
import os
from dataclasses import dataclass
from typing import Any, Dict, List, Union
from datasets import Dataset, DatasetDict
from transformers import (
    WhisperForConditionalGeneration,
    WhisperProcessor,
    Seq2SeqTrainingArguments,
    Seq2SeqTrainer,
)

# --- 1. Конфигурация и загрузка данных ---
MODEL_NAME = "openai/whisper-large-v3"
OUTPUT_DIR = "../../models/whisper-fine-tuned-a100"
DATA_CSV_PATH = "../../data/clear_rez_transcriptions.csv"
CPU_CORES = 60

print("Загрузка и подготовка данных...")
df = pd.read_csv(DATA_CSV_PATH)
df = df[["Путь к файлу", "Расшифровка", "test"]].dropna()
df.rename(columns={"Путь к файлу": "audio", "Расшифровка": "sentence"}, inplace=True)
df['audio'] = df['audio'].apply(lambda path: f"../../data/{path}")

# --- ИЗМЕНЕНИЕ: Разделение данных по столбцу 'test' ---
print("Разделение данных на обучающую и валидационную выборки...")
train_df = df[df['test'] == 0].copy()
eval_df = df[df['test'] == 1].copy()

# Создаем DatasetDict
raw_datasets = DatasetDict()
raw_datasets["train"] = Dataset.from_pandas(train_df)
raw_datasets["eval"] = Dataset.from_pandas(eval_df)

print(f"Размер обучающей выборки: {len(raw_datasets['train'])}")
print(f"Размер валидационной выборки: {len(raw_datasets['eval'])}")

# --- 2. Загрузка процессора и подготовка датасета ---
print("Загрузка процессора...")
processor = WhisperProcessor.from_pretrained(MODEL_NAME, language="russian", task="transcribe")

def prepare_dataset(batch):
    audio = batch["audio"]
    batch["input_features"] = processor(audio["array"], sampling_rate=16_000).input_features[0]
    batch["labels"] = processor.tokenizer(batch["sentence"]).input_ids
    return batch

print(f"Обработка датасетов с использованием {CPU_CORES} ядер CPU...")
# Приводим аудиоколонку к нужному типу перед map
raw_datasets = raw_datasets.cast_column("audio", Audio(sampling_rate=16_000))
tokenized_datasets = raw_datasets.map(prepare_dataset, remove_columns=raw_datasets["train"].column_names, num_proc=CPU_CORES)


# --- 3. Data Collator и метрики ---
@dataclass
class DataCollatorSpeechSeq2SeqWithPadding:
    processor: Any
    def call(self, features: List[Dict[str, Union[List[int], torch.Tensor]]]) -> Dict[str, torch.Tensor]:
        input_features = [{"input_features": feature["input_features"]} for feature in features]
        batch = self.processor.feature_extractor.pad(input_features, return_tensors="pt")
        label_features = [{"input_ids": feature["labels"]} for feature in features]
        labels_batch = self.processor.tokenizer.pad(label_features, return_tensors="pt")
        labels = labels_batch["input_ids"].masked_fill(labels_batch.attention_mask.ne(1), -100)
        if (labels[:, 0] == self.processor.tokenizer.bos_token_id).all().cpu().item():
            labels = labels[:, 1:]
        batch["labels"] = labels
        return batch

data_collator = DataCollatorSpeechSeq2SeqWithPadding(processor=processor)
wer_metric = evaluate.load("wer")

def compute_metrics(pred):
    pred_ids, label_ids = pred.predictions, pred.label_ids
    label_ids[label_ids == -100] = processor.tokenizer.pad_token_id
    pred_str = processor.tokenizer.batch_decode(pred_ids, skip_special_tokens=True)
    label_str = processor.tokenizer.batch_decode(label_ids, skip_special_tokens=True)
    wer = 100 * wer_metric.compute(predictions=pred_str, references=label_str)
    return {"wer": wer}

# --- 4. Обучение ---
print("Загрузка модели для дообучения...")
model = WhisperForConditionalGeneration.from_pretrained(MODEL_NAME)
model.config.forced_decoder_ids = None
model.config.suppress_tokens = []

training_args = Seq2SeqTrainingArguments(
    output_dir=OUTPUT_DIR,
    per_device_train_batch_size=16,
    per_device_eval_batch_size=16,
    bf16=True,
    learning_rate=1e-5,
    warmup_steps=500,
    max_steps=2500,
    evaluation_strategy="steps",
    predict_with_generate=True,
    generation_max_length=225,
    save_steps=500,       # <--- Сохраняем чекпоинт каждые 500 шагов
    save_total_limit=3,   # <--- Храним до 3 последних чекпоинтов
    eval_steps=500,
    logging_steps=50,
    report_to=["tensorboard"],
    load_best_model_at_end=True,
    metric_for_best_model="wer",
    greater_is_better=False,
    optim="adafactor",
)

trainer = Seq2SeqTrainer(
    args=training_args,
    model=model,
    train_dataset=tokenized_datasets["train"],
    eval_dataset=tokenized_datasets["eval"],
    data_collator=data_collator,
    compute_metrics=compute_metrics,
    tokenizer=processor.feature_extractor,
)

# --- Логика возобновления обучения ---
# Проверяем, существуют ли чекпоинты в директории вывода
last_checkpoint = None
if os.path.isdir(training_args.output_dir):
    # Ищем последнюю папку checkpoint-XXXX
    checkpoints = [d for d in os.listdir(training_args.output_dir) if d.startswith("checkpoint-")]
    if checkpoints:
        last_checkpoint = max(checkpoints, key=lambda x: int(x.split('-')[-1]))
        last_checkpoint_path = os.path.join(training_args.output_dir, last_checkpoint)
        print(f"Найден чекпоинт: {last_checkpoint_path}. Обучение будет возобновлено с него.")
        
print("\n--- Запуск дообучения Whisper ---\n")

# trainer.train() автоматически подхватит последний чекпоинт,
# если он был найден в output_dir.
# Если нужно явно указать, можно передать resume_from_checkpoint=last_checkpoint_path
trainer.train(resume_from_checkpoint=last_checkpoint_path if last_checkpoint else None)

print("\n--- Обучение завершено. Сохранение итоговой модели... ---\n")

trainer.save_model(OUTPUT_DIR)
processor.save_pretrained(OUTPUT_DIR)

print(f"Модель и процессор сохранены в {OUTPUT_DIR}")
