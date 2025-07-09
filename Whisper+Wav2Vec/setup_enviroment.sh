bash
#!/bin/bash

echo "Шаг 1/2: Установка PyTorch для CUDA 11.8..."
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118

echo "Шаг 2/2: Установка остальных зависимостей из requirements.txt..."
pip install -r requirements.txt

echo "Среда успешно настроена."