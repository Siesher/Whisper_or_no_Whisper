# data_utils.py
import pandas as pd
import re
from pathlib import Path
from sklearn.model_selection import train_test_split
from datasets import Dataset, load_dataset
from tqdm import tqdm
from faster_whisper import WhisperModel as FasterWhisperModel
import wandb
import torch

def normalize_text(text):
    if not isinstance(text, str): return ""
    text = text.lower()
    text = re.sub(r'[^\w\s]', '', text)
    text = re.sub(r'\s+', ' ', text).strip()
    return text

def prepare_asr_data(config):
    print("\n--- ЭТАП 1.1: ПОДГОТОВКА ДАННЫХ ДЛЯ ASR ---")
    paths_cfg = config['paths']
    data_cfg = config['data_prep']
    
    df = pd.read_csv(paths_cfg['source_csv'])
    new_audio_dir = Path(paths_cfg['audio_directory'])
    df['corrected_path'] = df[data_cfg['path_column']].apply(lambda p: str(new_audio_dir / Path(p).name))
    df['normalized_text'] = df[data_cfg['text_column']].apply(normalize_text)
    df = df.dropna(subset=['normalized_text'])
    df = df[df['normalized_text'].str.len() > 0]
    df = df[df['corrected_path'].apply(lambda p: Path(p).exists())]
    
    train_df, test_df = train_test_split(df, test_size=data_cfg['test_size'], random_state=42)
    
    prepared_dir = Path(paths_cfg['prepared_data_dir'])
    prepared_dir.mkdir(exist_ok=True)
    
    # ... (создание JSONL и манифестов для Silero, как в предыдущей версии) ...

    print("Логирование датасета в W&B Artifacts...")
    run = wandb.init(project=config['wandb']['project'], entity=config['wandb']['entity'], job_type="data_preparation")
    artifact = wandb.Artifact('asr_datasets', type='dataset')
    artifact.add_dir(str(prepared_dir))
    run.log_artifact(artifact)
    run.finish()
    print("Данные для ASR подготовлены и залогированы.")

def prepare_llm_dpo_data(config):
    print("\n--- ЭТАП 1.2: ПОДГОТОВКА ДАННЫХ ДЛЯ DPO LLM ---")
    paths_cfg = config['paths']
    llm_cfg = config['yandex_gpt_params']
    
    train_df = pd.read_json(Path(paths_cfg['prepared_data_dir']) / "whisper_train.jsonl", lines=True)
    
    print("Генерация 'плохих' транскрипций с базовой моделью Whisper...")
    base_whisper = FasterWhisperModel("openai/whisper-large-v3", device="cuda", compute_type="float16")
    
    dpo_data = []
    for _, row in tqdm(train_df.iterrows(), total=len(train_df), desc="Создание DPO пар"):
        segments, _ = base_whisper.transcribe(row['audio'], language="ru")
        rejected_response = "".join(s.text for s in segments).strip()
        
        # Для DPO нам нужна тройка: (промпт, хороший ответ, плохой ответ)
        dpo_data.append({
            "prompt": llm_cfg['dpo_prompt_template'].format(instruction=rejected_response),
            "chosen": row['transcription'],
            "rejected": rejected_response,
        })
    
    dpo_dataset = Dataset.from_list(dpo_data)
    dpo_dataset.save_to_disk(paths_cfg['llm_dpo_data_path'])
    
    print("Логирование DPO датасета в W&B Artifacts...")
    run = wandb.init(project=config['wandb']['project'], entity=config['wandb']['entity'], job_type="dpo_data_prep")
    artifact = wandb.Artifact('dpo_dataset', type='dataset')
    artifact.add_dir(paths_cfg['llm_dpo_data_path'])
    run.log_artifact(artifact)
    run.finish()
    print("Данные для DPO подготовлены и залогированы.")