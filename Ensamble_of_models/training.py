# training.py
# (Содержит функции run_whisper_finetune и run_silero_finetune)
# Они адаптированы для работы с W&B Artifacts для загрузки данных и сохранения моделей.
# Здесь для краткости приведена только функция для Whisper.

import wandb
import ctranslate2
from transformers import Seq2SeqTrainer, Seq2SeqTrainingArguments, WhisperForConditionalGeneration, WhisperProcessor
from datasets import load_dataset, Audio
from pathlib import Path

def run_whisper_finetune(config):
    print("\n--- ЭТАП 3.1: ДООБУЧЕНИЕ И ОПТИМИЗАЦИЯ WHISPER ---")
    w_cfg = config['whisper_params']
    paths_cfg = config['paths']
    
    run = wandb.init(project=config['wandb']['project'], entity=config['wandb']['entity'], job_type="train_whisper")
    
    print("Загрузка ASR датасета из W&B Artifacts...")
    artifact = run.use_artifact('asr_datasets:latest')
    artifact_dir = Path(artifact.download())
    
    dataset = load_dataset("json", data_files={
        'train': str(artifact_dir / "whisper_train.jsonl"),
        'test': str(artifact_dir / "whisper_test.jsonl")
    }).cast_column("audio", Audio(sampling_rate=16000))
    
    # ... (остальной код подготовки датасета, модели и трейнера, как в прошлом скрипте) ...
    # ... с bf16=True и report_to="wandb" ...
    
    print("Обучение Whisper завершено!")
    
    print("Сохранение и логирование модели Whisper в W&B Artifacts...")
    model_artifact = wandb.Artifact(f"whisper-{run.id}", type="model")
    # ... (код сохранения модели в папку) ...
    model_artifact.add_dir(paths_cfg['whisper_finetuned_path'])
    run.log_artifact(model_artifact)

    # ... (код конвертации в CTranslate2) ...
    print("Конвертация в CTranslate2 завершена.")
    run.finish()

def run_silero_finetune(config):
    # Аналогично, но с логикой для Silero.
    print("\n--- ЭТАП 3.2: ДООБУЧЕНИЕ SILERO ---")
    print("Обучение Silero (пропущено для демонстрации).")