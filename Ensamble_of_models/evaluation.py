# evaluation.py
# (Содержит логику оценки)
import wandb
from pathlib import Path
import pandas as pd
from vllm import LLM, SamplingParams # Используем vLLM
from tqdm import tqdm
import evaluate

def run_advanced_evaluation(config):
    print("\n--- ЭТАП 4: ФИНАЛЬНАЯ ОЦЕНКА ---")
    paths_cfg = config['paths']
    
    run = wandb.init(project=config['wandb']['project'], entity=config['wandb']['entity'], job_type="evaluation")
    
    # ... (Загрузка ASR моделей и тестового датасета через wandb.use_artifact) ...

    print("Загрузка дообученной LLM для vLLM...")
    llm = LLM(
        model=paths_cfg['llm_finetuned_merged_path'],
        tensor_parallel_size=config['vllm_params']['tensor_parallel_size']
    )
    sampling_params = SamplingParams(temperature=0.7, top_p=0.95, max_tokens=512)

    prompts = []
    references = []
    # ... (Цикл по тестовому датасету, где вы генерируете whisper_text_raw и добавляете в список prompts)

    print("Запуск батч-инференса LLM с vLLM...")
    llm_outputs = llm.generate(prompts, sampling_params)
    
    # ... (далее цикл по результатам, ROVER, подсчет всех WER и логирование в wandb) ...
    run.finish()