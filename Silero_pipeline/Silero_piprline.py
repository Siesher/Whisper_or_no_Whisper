import os
import torch
import time
import yaml
import argparse
import pandas as pd
import re
import subprocess
import git
from urllib.request import urlretrieve
from sklearn.model_selection import train_test_split

# --- ВСПОМОГАТЕЛЬНЫЕ ФУНКЦИИ ---


def load_config(config_path):
    print(f"Загрузка конфигурации из {config_path}...")
    with open(config_path, 'r', encoding='utf-8') as f:
        return yaml.safe_load(f)


def setup_environment(config):
    """Клонирует репозиторий Silero и скачивает базовую модель, если их нет."""
    print("\n--- ЭТАП 1: НАСТРОЙКА ОКРУЖЕНИЯ ---")
    repo_path = config['environment']['silero_repo_path']
    if not os.path.exists(repo_path):
        print(f"Клонирование репозитория Silero из {config['environment']['silero_repo_url']}...")
        git.Repo.clone_from(config['environment']['silero_repo_url'], repo_path)
    else:
        print("Репозиторий Silero уже существует.")
    
    base_model_path = os.path.join(repo_path, os.path.basename(config['model']['base_model_url']))
    if not os.path.exists(base_model_path):
        print(f"Скачивание базовой модели {os.path.basename(config['model']['base_model_url'])}...")
        urlretrieve(config['model']['base_model_url'], base_model_path)
    else:
        print("Базовая модель уже скачана.")
    return repo_path, base_model_path


def prepare_data_for_silero(config, repo_path):
    """Читает JSON, очищает текст и создает train.txt, test.txt, chars.txt."""
    print("\n--- ЭТАП 2: ПОДГОТОВКА ДАННЫХ ДЛЯ SILERO ---")
    df = pd.read_json(config['paths']['data_json'])
    
    # Очистка текста: нижний регистр, только кириллица и пробелы
    chars_to_remove_regex = '[^\u0400-\u04FF\s]'
    df['text'] = df['text'].apply(lambda x: re.sub(chars_to_remove_regex, '', x).lower())
    df = df.dropna()

    # Разделение на train/test
    train_df, test_df = train_test_split(df, test_size=config['data_prep']['test_size'])
    
    # Создание папки для данных
    data_dir = os.path.join(repo_path, config['data_prep']['prepared_data_dir'])
    os.makedirs(data_dir, exist_ok=True)

    # Запись train.txt и test.txt
    def write_manifest(dataframe, path):
        with open(path, 'w', encoding='utf-8') as f:
            for _, row in dataframe.iterrows():
                # Пути должны быть абсолютными для надежности
                audio_path = os.path.abspath(row['audio_path'])
                f.write(f"{audio_path}|{row['text']}\n")
    
    train_path = os.path.join(data_dir, 'train.txt')
    test_path = os.path.join(data_dir, 'test.txt')
    write_manifest(train_df, train_path)
    write_manifest(test_df, test_path)
    print(f"Файлы train.txt и test.txt созданы в {data_dir}")

    # Создание chars.txt (алфавит)
    all_text = "".join(df['text'].tolist())
    unique_chars = sorted(list(set(all_text)))
    chars_path = os.path.join(data_dir, 'chars.txt')
    with open(chars_path, 'w', encoding='utf-8') as f:
        f.write('_\n') # Пустой символ CTC
        f.write('\n'.join(unique_chars))
        f.write('\n \n') # Пробел
    print(f"Файл chars.txt создан в {data_dir}")
    
    return train_path, test_path, chars_path


def generate_silero_config(config, repo_path, base_model_path, train_path, test_path, chars_path):
    """Генерирует YAML-конфиг, который понимает скрипт обучения Silero."""
    silero_config = {
        'train_path': os.path.abspath(train_path),
        'test_path': os.path.abspath(test_path),
        'chars_path': os.path.abspath(chars_path),
        'pretrain_path': os.path.abspath(base_model_path),
        'work_dir': config['training']['work_dir'],
        'batch_size': config['training']['batch_size'],
        'num_epochs': config['training']['num_epochs'],
        'learning_rate': config['training']['learning_rate'],
        # Стандартные параметры для v4 модели, можно вынести в конфиг при необходимости
        'model': {
            'in_channels': 1,
            'out_channels': 1024,
            'conv1': {'out_channels': 256},
            'conv_blocks': [
                {'out_channels': 256}, {'out_channels': 256}, {'out_channels': 256},
                {'out_channels': 512}, {'out_channels': 512}, {'out_channels': 512},
                {'out_channels': 512}
            ],
            'conv_last': {'out_channels': 1024}
        }
    }
    
    config_save_path = os.path.join(repo_path, 'generated_finetune_config.yaml')
    with open(config_save_path, 'w', encoding='utf-8') as f:
        yaml.dump(silero_config, f, default_flow_style=False, allow_unicode=True)
        
    print(f"Конфигурационный файл для Silero сгенерирован: {config_save_path}")
    return config_save_path


def run_training(repo_path, silero_config_path):
    """Запускает скрипт train.py из репозитория Silero."""
    print("\n--- ЭТАП 3: ЗАПУСК ПРОЦЕССА ДООБУЧЕНИЯ SILERO ---")
    train_script_path = os.path.join(repo_path, 'train.py')
    # Используем относительный путь для --config-path, так как CWD будет репозиторий
    config_name = os.path.basename(silero_config_path)

    # Команда для запуска. Важно указать `cwd=repo_path`
    command = [
        "python", train_script_path,
        f"--config-path=.",
        f"--config-name={config_name}"
    ]
    
    print(f"Выполнение команды: {' '.join(command)}")
    print(f"Рабочая директория: {os.path.abspath(repo_path)}")
    
    try:
        # Запускаем процесс и ждем его завершения
        result = subprocess.run(command, cwd=repo_path, check=True, text=True, capture_output=True)
        print("STDOUT:", result.stdout)
        print("STDERR:", result.stderr)
        print("Обучение успешно завершено!")
    except subprocess.CalledProcessError as e:
        print("Ошибка во время обучения!")
        print("Код возврата:", e.returncode)
        print("STDOUT:", e.stdout)
        print("STDERR:", e.stderr)
        return False
    return True


def final_evaluation(config, repo_path):
    """Демонстрирует, как использовать дообученную модель."""
    # Этот этап будет демонстрационным, так как полная оценка требует отдельного скрипта
    print("\n--- ЭТАП 4: ДЕМОНСТРАЦИЯ ИСПОЛЬЗОВАНИЯ МОДЕЛИ ---")
    work_dir = os.path.join(repo_path, config['training']['work_dir'])
    final_model_path = os.path.join(work_dir, 'latest.pt') # Silero часто сохраняет так
    if not os.path.exists(final_model_path):
        final_model_path = os.path.join(work_dir, config['model']['finetuned_model_name'])
        if not os.path.exists(final_model_path):
           print(f"Не удалось найти финальную модель в {work_dir}. Пропускаем оценку.")
           return
           
    print(f"Предполагается, что ваша финальная модель находится здесь: {final_model_path}")
    print("Вы можете использовать эту модель в своем коде, загрузив ее через PyTorch,")
    print("как было показано в наших предыдущих обсуждениях.")
    # Примерный код для инференса можно добавить сюда по желанию.


def main():
    parser = argparse.ArgumentParser(description="End-to-end Silero ASR model fine-tuning pipeline.")
    parser.add_argument("--config", type=str, required=True, help="Path to the YAML configuration file.")
    args = parser.parse_args()
    
    config = load_config(args.config)
    
    repo_path, base_model_path = setup_environment(config)
    train_path, test_path, chars_path = prepare_data_for_silero(config, repo_path)
    silero_config_path = generate_silero_config(config, repo_path, base_model_path, train_path, test_path, chars_path)
    
    success = run_training(repo_path, silero_config_path)
    if success:
        final_evaluation(config, repo_path)


if __name__ == "__main__":
    main()