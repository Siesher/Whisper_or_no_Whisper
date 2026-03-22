<div align="center">

<img src="figures/banner.svg" alt="Whisper or no Whisper Banner" width="100%">

# 🎙️ Whisper or no Whisper

_«Точность распознавания — это не роскошь, а требование безопасности.»_

**Коллекция production-grade пайплайнов для русскоязычного Speech-to-Text с fine-tuning, ансамблированием и LLM-коррекцией**

[![Python 3.12+](https://img.shields.io/badge/Python-3.12+-818CF8?style=for-the-badge&logo=python&logoColor=white)](https://python.org)
[![Whisper](https://img.shields.io/badge/Whisper-large--v3-C4B5FD?style=for-the-badge&logo=openai&logoColor=white)](https://huggingface.co/openai/whisper-large-v3)
[![YandexGPT](https://img.shields.io/badge/YandexGPT-5--Lite--8B-6366F1?style=for-the-badge&logo=yandex&logoColor=white)](https://huggingface.co/yandex-datasphere)
[![W&B](https://img.shields.io/badge/Weights_%26_Biases-Tracking-818CF8?style=for-the-badge&logo=weightsandbiases&logoColor=white)](https://wandb.ai)
[![License: MIT](https://img.shields.io/badge/License-MIT-C4B5FD?style=for-the-badge)](LICENSE)

[О проекте](#-о-проекте) · [Модули](#-модули) · [Архитектура](#-архитектура-ensemble) · [Быстрый старт](#-быстрый-старт) · [Результаты](#-метрики)

---

</div>

## ✦ О проекте

Набор пайплайнов для распознавания русской речи, оптимизированных для специализированных доменов (железнодорожная диспетчерская связь). Проект эволюционирует от single-model подходов к production-grade ансамблю с LLM-коррекцией.

| **Ключевые особенности** | **Описание** |
|---|---|
| 🎯 4 независимых пайплайна | От простого fine-tuning до multi-model ensemble |
| 🗳️ Ensemble voting | ROVER + YandexGPT для объединения гипотез |
| ⚡ CTranslate2 оптимизация | int8/float16 квантизация для быстрого инференса |
| 🧪 DPO fine-tuning LLM | YandexGPT как корректор ошибок ASR |
| 📊 W&B интеграция | Tracking экспериментов и версионирование артефактов |
| 🖥️ Multi-GPU | Accelerate + vLLM tensor parallelism (2×A100) |

---

## ✦ Модули

Проект содержит 4 модуля возрастающей сложности:

```
Whisper_or_no_Whisper/
│
├── 1. Whisper_v3_large/         Базовый fine-tuning Whisper
├── 2. Silero_pipeline/          Автоматизированный fine-tuning Silero
├── 3. Whisper+Wav2Vec/          Dual-model ensemble + YandexGPT
└── 4. Ensamble_of_models/       Production pipeline: 3 модели + DPO + ROVER
```

| # | Модуль | Модели | Назначение |
|---|---|---|---|
| 1 | **Whisper_v3_large** | Whisper-large-v3 | Single-model baseline + CTranslate2 |
| 2 | **Silero_pipeline** | Silero v4 | Лёгкий русскоязычный ASR |
| 3 | **Whisper+Wav2Vec** | Whisper + Wav2Vec2-XLS-R-53 | Dual-model + YandexGPT chat merge |
| 4 | **Ensamble_of_models** | Whisper + Silero + YandexGPT | Full pipeline: training → DPO → ROVER voting |

---

## ✦ Архитектура Ensemble

### Модуль 4: Ensamble_of_models (Production)

```
 ╔═══════════════════════════════════════════════════════════╗
 ║  Stage 1: Data Preparation                               ║
 ║  CSV → normalize → train/test split → manifests          ║
 ║  + generate DPO pairs (good/bad transcriptions)           ║
 ╚═════════════════════════╤═════════════════════════════════╝
                           ▼
 ┌─────────────────────────────────────────────────────────┐
 │  Stage 2: Parallel Model Training                       │
 │                                                         │
 │  ┌──────────────┐  ┌──────────────┐  ┌──────────────┐  │
 │  │   Whisper     │  │    Silero    │  │  YandexGPT   │  │
 │  │  large-v3     │  │    v4_ru     │  │  5-Lite-8B   │  │
 │  │              │  │              │  │              │  │
 │  │  HF Trainer  │  │  Official    │  │  QLoRA + DPO │  │
 │  │  + CTranslate│  │  train.py    │  │  (corrector) │  │
 │  └──────┬───────┘  └──────┬───────┘  └──────┬───────┘  │
 └─────────┼─────────────────┼─────────────────┼──────────┘
           ▼                 ▼                 ▼
 ╔═══════════════════════════════════════════════════════════╗
 ║  Stage 3: Ensemble Inference                             ║
 ║                                                          ║
 ║  Audio ──→ Whisper hypothesis ──┐                        ║
 ║        ──→ Silero hypothesis  ──┼──→ ROVER voting ──→ WER║
 ║        ──→ YandexGPT correction─┘    (vLLM batch)       ║
 ╚══════════════════════════════════════════════════════════╝
```

### Модуль 3: Whisper+Wav2Vec (Dual Ensemble)

```
 Audio ──→ Whisper (GPU 0) ──→ Transcription A ──┐
       ──→ Wav2Vec2 (GPU 1) ──→ Transcription B ──┼──→ YandexGPT Chat ──→ Output
                                                   │    (merge + correct)
                                                   └─────────────────────
```

---

## ✦ Структура проекта

```
Whisper_or_no_Whisper/
├── Ensamble_of_models/                # Production ensemble pipeline
│   ├── main.py                        #   Entry point (checkpoint-based)
│   ├── config.yaml                    #   Master configuration
│   ├── data_utils.py                  #   Data prep + DPO pair generation
│   ├── training.py                    #   Whisper & Silero fine-tuning
│   ├── evaluation.py                  #   Ensemble eval + vLLM inference
│   ├── llm_funetuning.py             #   YandexGPT QLoRA + DPO
│   ├── requirements.txt              #   16 dependencies
│   └── *.ipynb                        #   Notebook versions
│
├── Silero_pipeline/                   # Silero fine-tuning orchestrator
│   ├── Silero_piprline.py             #   Auto-clone, auto-config, auto-train
│   ├── config.yaml                    #   Silero parameters
│   └── requirements.txt
│
├── Whisper_v3_large/                  # Whisper baseline + optimization
│   ├── Whisper_large_v3_turbo.py      #   Fine-tune + CTranslate2
│   ├── Whisper.ipynb                  #   Notebook version
│   ├── config.yaml                    #   Training configuration
│   └── requirements.txt
│
├── Whisper+Wav2Vec/                   # Dual-model ensemble
│   ├── scripts/
│   │   ├── fine_tune_whisper.py       #   Multi-GPU Whisper training
│   │   ├── fine_tune_Wav2Vec2.py      #   Wav2Vec2 CTC training
│   │   └── run_ensemble_models.py     #   Ensemble inference + YandexGPT
│   ├── setup_enviroment.sh            #   Environment setup
│   └── requirements.txt
│
├── figures/
│   └── banner.svg                     # README banner
└── README.md
```

---

## ✦ Быстрый старт

### Требования

| Компонент | Версия | Примечание |
|---|---|---|
| **Python** | 3.12+ | Все пайплайны |
| **PyTorch** | 2.0+ | CUDA 11.8 / 12.1 |
| **VRAM** | 24+ GB | Single GPU (gradient accumulation) |
| **Оптимально** | 2×A100 40GB | Для ensemble + vLLM inference |

### Модуль 1: Whisper (baseline)

```bash
cd Whisper_v3_large
pip install -r requirements.txt
python Whisper_large_v3_turbo.py --config config.yaml
```

### Модуль 2: Silero

```bash
cd Silero_pipeline
pip install -r requirements.txt
python Silero_piprline.py --config config.yaml
```

### Модуль 3: Whisper + Wav2Vec2

```bash
cd Whisper+Wav2Vec
bash setup_enviroment.sh

# Training (multi-GPU)
accelerate launch scripts/fine_tune_whisper.py
accelerate launch scripts/fine_tune_Wav2Vec2.py

# Ensemble inference
python scripts/run_ensemble_models.py
```

### Модуль 4: Production Ensemble

```bash
cd Ensamble_of_models
pip install -r requirements.txt
python main.py --config config.yaml
```

Pipeline управляется через флаги в `config.yaml`:

```yaml
pipeline:
  run_data_preparation: true
  run_whisper_training: true
  run_silero_training: true
  run_llm_dpo_data_prep: true
  run_llm_finetuning: true
  run_evaluation: true
```

---

## ✦ Конфигурация моделей

### Whisper Fine-tuning

| Параметр | Значение |
|---|---|
| **Base model** | `openai/whisper-large-v3` |
| **Learning rate** | 1e-5 |
| **Batch size** | 16 (per GPU) |
| **Gradient accumulation** | 2 |
| **Max steps** | 2000 |
| **Warmup** | 200 steps |
| **Precision** | bf16 |
| **Optimization** | CTranslate2 (int8 / float16) |

### YandexGPT DPO

| Параметр | Значение |
|---|---|
| **Base model** | `yandex-datasphere/yandex-gpt-5-lite-8b-pretrain` |
| **LoRA rank** | 16 |
| **LoRA alpha** | 32 |
| **Quantization** | 4-bit (QLoRA) |
| **DPO beta** | 0.1 |
| **Learning rate** | 2e-5 |
| **Epochs** | 2 |

### Silero

| Параметр | Значение |
|---|---|
| **Base model** | `v4_ru.pt` |
| **Batch size** | 64 |
| **Epochs** | 20 |
| **Learning rate** | 1e-4 |

---

## ✦ Метрики

| Метрика | Описание |
|---|---|
| **WER** (Word Error Rate) | Основная метрика качества транскрипции |
| **Evaluation** | Side-by-side: base vs fine-tuned vs optimized vs ensemble |
| **ROVER voting** | Recognition Output Voting Error Reduction |

---

## ✦ Технологии

| Layer | Stack |
|---|---|
| **ASR Models** | Whisper-large-v3 · Wav2Vec2-XLS-R-53 · Silero v4 |
| **LLM Corrector** | YandexGPT-5-Lite-8B (QLoRA + DPO) |
| **Training** | HuggingFace Trainer · Accelerate · PEFT · TRL |
| **Optimization** | CTranslate2 · vLLM (tensor parallelism) · bitsandbytes |
| **Audio** | librosa · soundfile · torchaudio |
| **Tracking** | Weights & Biases (experiments + artifacts) |
| **Metrics** | jiwer (WER) · ROVER |

---

## ✦ Data Pipeline

```
 CSV (audio paths + transcriptions)
         │
         ▼
 ┌──────────────────────┐
 │  Normalize text       │ ← lowercase, remove punctuation,
 │  + train/test split   │   deduplicate whitespace
 └──────────┬───────────┘
            ▼
 ┌──────────────────────┐
 │  Prepare manifests    │ ← JSONL (Silero), HF Dataset (Whisper)
 └──────────┬───────────┘
            ▼
 ┌──────────────────────┐
 │  Generate DPO pairs   │ ← base Whisper → "bad" transcriptions
 │  (prompt, chosen,     │   ground truth → "good" transcriptions
 │   rejected)           │
 └──────────────────────┘
```

**Формат входных данных** (CSV):

| Путь к файлу | Расшифровка | test |
|---|---|---|
| `/data/audio_001.wav` | транскрипция текста | 0 |
| `/data/audio_002.wav` | другой текст | 1 |

---

## ✦ Лицензия

MIT License — свободно используйте, форкайте, дорабатывайте.

---

<div align="center">

**Сухацкий Максим** · МГТУ им. Н.Э. Баумана (Калужский филиал) · 2025

[![HuggingFace](https://img.shields.io/badge/%F0%9F%A4%97_HuggingFace-Siesher-C4B5FD?style=flat-square)](https://huggingface.co/Siesher)
[![GitHub](https://img.shields.io/badge/GitHub-Siesher-818CF8?style=flat-square&logo=github)](https://github.com/Siesher)

</div>
