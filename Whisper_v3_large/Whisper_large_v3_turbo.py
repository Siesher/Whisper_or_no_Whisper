import os
import torch
import time
import yaml
import argparse
from datasets import load_dataset, Audio
from transformers import (
    WhisperForConditionalGeneration,
    WhisperProcessor,
    Seq2SeqTrainingArguments,
    Seq2SeqTrainer,
    DataCollatorForSeq2SeqWithPadding
)
import evaluate
import ctranslate2


def load_config(config_path):
    """Загружает конфигурацию из YAML файла."""
    print(f"Загрузка конфигурации из {config_path}...")
    with open(config_path, 'r') as f:
        return yaml.safe_load(f)


def prepare_data(config):
    """Загружает и подготавливает датасет для обучения."""
    print("\n--- ЭТАП 1: ПОДГОТОВКА ДАННЫХ ---")
    dataset = load_dataset("json", data_files=config['paths']['data_json'], 
                           split="train")
    dataset = dataset.rename_column("audio_path", "audio")
    dataset = dataset.rename_column("text", "transcription")
    dataset = dataset.cast_column("audio", Audio(sampling_rate=16000))
    dataset = dataset.train_test_split(test_size=0.1)

    processor = WhisperProcessor.from_pretrained(
        config['model']['base_model'],
        language=config['model']['language'],
        task=config['model']['task']
    )

    def prepare_fn(batch):
        audio = batch["audio"]
        batch["input_features"] = processor.feature_extractor(audio["array"], 
                                                              sampling_rate=16000).input_features[0]
        batch["labels"] = processor.tokenizer(batch["transcription"]).input_ids
        return batch

    print("Обработка и токенизация датасета...")
    tokenized_dataset = dataset.map(prepare_fn, 
                                    remove_columns=dataset.column_names["train"], num_proc=os.cpu_count())
    return dataset, tokenized_dataset, processor


def train_model(config, processor, tokenized_dataset, output_dir_finetuned):
    """Настраивает и запускает процесс дообучения модели."""
    print("\n--- ЭТАП 2: ДООБУЧЕНИЕ МОДЕЛИ (FINE-TUNING) ---")
    data_collator = DataCollatorForSeq2SeqWithPadding(processor=processor)
    metric = evaluate.load("wer")

    def compute_metrics(pred):
        pred_ids = pred.predictions
        label_ids = pred.label_ids
        label_ids[label_ids == -100] = processor.tokenizer.pad_token_id
        pred_str = processor.batch_decode(pred_ids, skip_special_tokens=True)
        label_str = processor.batch_decode(label_ids, skip_special_tokens=True)
        wer = 100 * metric.compute(predictions=pred_str, references=label_str)
        return {"wer": wer}

    model = WhisperForConditionalGeneration.from_pretrained(config['model']
                                                            ['base_model'])
    model.config.forced_decoder_ids = processor.get_decoder_prompt_ids(
        language=config['model']['language'], task=config['model']['task']
    )

    training_args = Seq2SeqTrainingArguments(output_dir=output_dir_finetuned,
                                             **config['training']['args'])

    trainer = Seq2SeqTrainer(
        args=training_args,
        model=model,
        train_dataset=tokenized_dataset["train"],
        eval_dataset=tokenized_dataset["test"],
        data_collator=data_collator,
        compute_metrics=compute_metrics,
        tokenizer=processor.feature_extractor,
    )

    print("Запуск процесса обучения...")
    trainer.train()
    print("Обучение завершено!")
    trainer.save_model(output_dir_finetuned)
    processor.save_pretrained(output_dir_finetuned)
    print(f"Дообученная PyTorch модель сохранена в: {output_dir_finetuned}")


def optimize_model(output_dir_finetuned, output_dir_optimized,
                   quantization_level):
    """Конвертирует и квантизует модель в формат CTranslate2."""
    print("\n--- ЭТАП 3: ОПТИМИЗАЦИЯ МОДЕЛИ ---")
    try:
        converter = ctranslate2.converters.TransformersConverter(
            output_dir_finetuned,
            copy_files=["preprocessor_config.json", "tokenizer.json",
                        "vocab.json"]
        )
        output_path = converter.convert(output_dir_optimized,
                    quantization=quantization_level, force=True)
        print(f"Оптимизированная модель сохранена в: {output_path}")
    except Exception as e:
        print(f"Ошибка при конвертации: {e}")


def final_evaluation(config, dataset, output_dir_finetuned, 
                     output_dir_optimized):
    
    """Проводит финальное тестирование и сравнивает производительность моделей."""
    
    print("\n--- ЭТАП 4: ФИНАЛЬНОЕ ТЕСТИРОВАНИЕ И СРАВНЕНИЕ ---")
    from insanely_fast_whisper import pipeline
    from transformers import pipeline as hf_pipeline

    test_sample = dataset["test"][0]
    test_audio_path = test_sample["audio"]["path"]
    print(f"Тестовый файл: {test_audio_path}")
    print(f"Оригинальный текст: {test_sample['transcription']}")

    device = "cuda:0" if torch.cuda.is_available() else "cpu"
    torch_dtype = torch.float16 if torch.cuda.is_available() else torch.float32

    # 1. Тест дообученной, но НЕ оптимизированной модели
    print("\n1. Тестируем дообученную модель (Hugging Face)...")
    hf_pipe = hf_pipeline("automatic-speech-recognition", 
                          model=output_dir_finetuned, device=device)
    start_time = time.time()
    hf_result = hf_pipe(test_audio_path, 
                        generate_kwargs={"language": config['model']['language']})
    hf_time = time.time() - start_time
    print(f"   Результат: {hf_result['text']}")
    print(f"   Время выполнения: {hf_time:.4f} сек")

    # 2. Тест ОПТИМИЗИРОВАННОЙ модели
    if os.path.exists(output_dir_optimized):
        print("\n2. Тестируем ОПТИМИЗИРОВАННУЮ модель (insanely-fast-whisper)...")
        optimized_pipe = pipeline(model_name=output_dir_optimized, 
                                  device=device, torch_dtype=torch_dtype)
        start_time = time.time()
        optimized_result = optimized_pipe(test_audio_path, batch_size=4)
        optimized_time = time.time() - start_time
        print(f"   Результат: {optimized_result['text']}")
        print(f"   Время выполнения: {optimized_time:.4f} сек")
        speedup = hf_time / optimized_time
        print(f"\nПрирост скорости: {speedup:.2f}x")
    else:
        print("\nОптимизированная модель не найдена. Пропускаем тест.")


def main():
    parser = argparse.ArgumentParser(description="End-to-end ASR model fine-tuning and optimization pipeline.")
    parser.add_argument("--config", type=str, required=True, help="Path to the YAML configuration file.")
    args = parser.parse_args()
    
    config = load_config(args.config)
    
    # Генерация путей для сохранения моделей
    model_name_suffix = config['model']['base_model'].split('/')[-1]
    output_dir_finetuned = f"./{model_name_suffix}-finetuned-ru"
    output_dir_optimized = f"{output_dir_finetuned}-ct2-{config['optimization']['quantization']}"
    
    # Запуск пайплайна
    dataset, tokenized_dataset, processor = prepare_data(config)
    train_model(config, processor, tokenized_dataset, output_dir_finetuned)
    
    optimize_model(output_dir_finetuned, 
                   output_dir_optimized, config['optimization']['quantization'])
    
    final_evaluation(config, dataset, 
                     output_dir_finetuned, output_dir_optimized)


if __name__ == "__main__":
    main()