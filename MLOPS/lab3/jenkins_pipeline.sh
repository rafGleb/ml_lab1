#!/bin/bash
set -e

echo "========================================"
echo "  JENKINS ML PIPELINE"
echo "========================================"

# Этап 1: Данные
echo -e "\n[STAGE 1/4] Получение данных..."
python3 download.py

# Этап 2: Обучение
echo -e "\n[STAGE 2/4] Обучение модели..."
python3 train_model.py

# Этап 3: Развертывание
echo -e "\n[STAGE 3/4] Развертывание модели..."
export BUILD_ID=dontKillMe
python3 deploy.py

# Этап 4: Проверка
echo -e "\n[STAGE 4/4] Проверка сервиса..."
sleep 3
bash health_check.sh

echo -e "\n========================================"
echo "  PIPELINE УСПЕШНО ЗАВЕРШЕН!"
echo "========================================"