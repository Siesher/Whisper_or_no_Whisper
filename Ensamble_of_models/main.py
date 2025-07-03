# main.py
import argparse
import yaml
import os
import subprocess
from pathlib import Path
import data_utils
import training

def load_config(config_path):
    with open(config_path, 'r', encoding='utf-8') as f: return yaml.safe_load(f)

def run_stage(stage_name, function, config, checkpoint_dir):
    checkpoint_file = checkpoint_dir / f"{stage_name}.done"
    if config['pipeline_control'][f'run_{stage_name}']:
        if checkpoint_file.exists():
            print(f"--- Пропуск этапа '{stage_name}' (найден чекпоинт) ---")
            return
        
        function(config)
        checkpoint_file.touch()
    else:
        print(f"--- Пропуск этапа '{stage_name}' (отключено в конфиге) ---")

def main():
    parser = argparse.ArgumentParser(description="Advanced ASR Ensembling Pipeline")
    parser.add_argument("--config", type=str, default="config.yaml")
    args = parser.parse_args()
    config = load_config(args.config)
    
    checkpoint_dir = Path(config['paths']['checkpoints_dir'])
    checkpoint_dir.mkdir(exist_ok=True)
    
    run_stage('data_prep', data_utils.prepare_asr_data, config, checkpoint_dir)
    run_stage('llm_dpo_data_prep', data_utils.prepare_llm_dpo_data, config, checkpoint_dir)

    # Здесь можно реализовать параллельный запуск
    run_stage('whisper_training', training.run_whisper_finetune, config, checkpoint_dir)
    run_stage('silero_training', training.run_silero_finetune, config, checkpoint_dir)
    
    # Запуск LLM fine-tuning через subprocess с accelerate
    if config['pipeline_control']['run_llm_finetuning']:
        checkpoint_file = checkpoint_dir / "llm_finetuning.done"
        if not checkpoint_file.exists():
            print("\n--- ЗАПУСК ДООБУЧЕНИЯ LLM ---")
            subprocess.run(["accelerate", "launch", "llm_finetuning.py", f"--config={args.config}"], check=True)
            checkpoint_file.touch()
        else:
            print("--- Пропуск этапа 'llm_finetuning' (найден чекпоинт) ---")

    # Импортируем только для последнего шага
    if config['pipeline_control']['run_evaluation']:
        import evaluation
        evaluation.run_advanced_evaluation(config)

    print("\nПайплайн завершил работу.")

if __name__ == "__main__":
    main()