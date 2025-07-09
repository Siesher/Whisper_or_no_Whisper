# Ансамбль STT моделей для распознавания речи диспетчеров

Этот проект представляет собой высокопроизводительную систему для распознавания речи (Speech-to-Text), оптимизированную для специфической задачи транскрибации переговоров железнодорожных диспетчеров.

Система использует ансамбль из трех моделей:
1.  Fine-tuned Whisper: Дообученная модель openai/whisper-large-v3 для максимальной точности на специфической лексике.
2.  Fine-tuned Wav2Vec2: Дообученная модель jonatasgrosman/wav2vec2-large-xlsr-53-russian в качестве второго "эксперта".
3.  YandexGPT-instruct: Локально развернутая модель yandex/YandexGPT-5-Lite-8B-instruct для финального объединения и редактирования результатов от первых двух моделей.

Проект оптимизирован для запуска на высокопроизводительном оборудовании (2x NVIDIA A100).

## Структура проекта

-   data/: Содержит исходный CSV файл с разметкой и папки с аудиофайлами.
-   models/: Директория для сохранения весов дообученных моделей.
-   scripts/training/: Скрипты для дообучения STT моделей.
-   scripts/inference/: Скрипт для запуска полного ансамбля на аудиофайле.
-   requirements.txt: Список необходимых Python библиотек.
-   setup_environment.sh: Bash-скрипт для быстрой установки окружения.
-   README.md: Этот файл.

## Пошаговая инструкция

### Шаг 1: Подготовка данных и окружения

1.  Клонируйте репозиторий и перейдите в его директорию.

2.  Разместите данные:
    -   Положите файл clear_rez_transcriptions.csv в папку data/.
    -   Все папки с аудио (Пульт ДСЦ, Пульт ДСП-6 и т.д.) также поместите внутрь папки data/.

3.  Создайте и активируйте виртуальное окружение:
   
    python -m venv venv
    source venv/bin/activate
    
4.  Запустите скрипт установки зависимостей. Он установит PyTorch для CUDA 11.8 и все остальные библиотеки.
   
    bash setup_environment.sh
    
### Шаг 2: Конфигурация для Multi-GPU

Для использования обеих A100 необходимо один раз сконфигурировать accelerate:

`bash
accelerate config

Ответьте на вопросы следующим образом:
-   In which computThis machinee you running?: **This machine**
-   Which multi-GPUne are you using?: **multi-GPU**
-   How many GPUs should be 2for distributed training?: **2**
-  nou want to use DeepSpeed?: **no**
-   Do you want tonoullyShardedDataParallel?: **no**
-   Do you want to unorch.distributed.launch`?: **no**
-   Wh(оставьте по умолчанию)se?: **(оставьте по умолчанию)**

### Шаг 3: Дообучение моделей

Запустите обучение для каждой модели. Этот процесс будет использовать обе GPU и может занять несколько часов.

1.  Дообучение Whisper:
   
    accelerate launch scripts/training/fine_tune_whisper.py
    
    Обученная модель сохранится в models/whisper-fine-tuned-a100/.

2.  Дообучение Wav2Vec2:
   
    accelerate launch scripts/training/fine_tune_wav2vec2.py
    
    Обученная модель сохранится в models/wav2vec2-fine-tuned-a100/.

### Шаг 4: Запуск ансамбля

После завершения обучения вы можете запустить основной скрипт для транскрибации аудиофайла. Он возьмет случайный файл из вашего датасета и выведет результаты.

python scripts/inference/run_ensemble.py
Первый запуск этого скрипта потребует времени на скачивание модели YandexGPT. Последующие запуски будут быстрее.