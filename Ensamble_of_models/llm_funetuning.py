# llm_finetuning.py
import torch
import wandb
from datasets import Dataset
from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig, TrainingArguments
from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training
from trl import DPOTrainer
from pathlib import Path
import yaml

def main():
    with open("config.yaml", 'r', encoding='utf-8') as f:
        config = yaml.safe_load(f)
    
    llm_cfg = config['yandex_gpt_params']
    paths_cfg = config['paths']
    wandb_cfg = config['wandb']

    run = wandb.init(project=wandb_cfg['project'], entity=wandb_cfg['entity'], job_type="train_llm_dpo")

    print("Загрузка DPO датасета из W&B Artifacts...")
    artifact = run.use_artifact('dpo_dataset:latest')
    dataset_path = artifact.download()
    dpo_dataset = Dataset.load_from_disk(dataset_path)

    print("Загрузка базовой LLM и токенизатора...")
    model_name = llm_cfg['base_model']
    tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
    tokenizer.pad_token = tokenizer.eos_token

    bnb_config = BitsAndBytesConfig(
        load_in_4bit=True, bnb_4bit_quant_type="nf4",
        bnb_4bit_compute_dtype=torch.bfloat16, bnb_4bit_use_double_quant=True
    )
    model = AutoModelForCausalLM.from_pretrained(
        model_name, quantization_config=bnb_config, device_map="auto", trust_remote_code=True
    )
    model.config.use_cache = False
    
    model = prepare_model_for_kbit_training(model)
    lora_config = LoraConfig(**llm_cfg['lora_config'])
    model = get_peft_model(model, lora_config)
    
    training_args = TrainingArguments(
        output_dir=paths_cfg['llm_finetuned_adapter_path'],
        report_to="wandb",
        **llm_cfg['dpo_training_args']
    )
    
    dpo_trainer = DPOTrainer(
        model, args=training_args,
        beta=0.1,
        train_dataset=dpo_dataset,
        tokenizer=tokenizer,
    )
    
    print("Запуск DPO fine-tuning...")
    dpo_trainer.train()
    
    print("Сохранение LoRA адаптера...")
    dpo_trainer.save_model(paths_cfg['llm_finetuned_adapter_path'])
    
    print("Объединение адаптера с базовой моделью для инференса...")
    merged_model = dpo_trainer.model.merge_and_unload()
    merged_model.save_pretrained(paths_cfg['llm_finetuned_merged_path'])
    tokenizer.save_pretrained(paths_cfg['llm_finetuned_merged_path'])
    
    print("Логирование финальной LLM в W&B Artifacts...")
    model_artifact = wandb.Artifact(f"yagpt-corrector-{run.id}", type="model")
    model_artifact.add_dir(paths_cfg['llm_finetuned_merged_path'])
    run.log_artifact(model_artifact)
    run.finish()

if __name__ == "__main__":
    main()