#!/usr/bin/env python3
"""
Этап 3: Развертывание модели через MLflow models serve
"""
import os
import sys
import subprocess
import time

def deploy_model():
    print("[INFO] Чтение пути к лучшей модели...")
    
    if not os.path.exists("best_model.txt"):
        print("[ERROR] Файл best_model.txt не найден!")
        sys.exit(1)
    
    with open("best_model.txt", "r") as f:
        model_path = f.read().strip()
    
    print(f"[INFO] Путь к модели: {model_path}")
    
    # Запускаем MLflow models serve в фоне
    print("[INFO] Запуск MLflow models serve на порту 5003...")
    
    cmd = f"mlflow models serve -m {model_path} -p 5003 --no-conda"
    
    process = subprocess.Popen(
        cmd.split(),
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL
    )
    
    # Ждем запуска
    time.sleep(10)
    
    print(f"[INFO] Сервис запущен в фоне (PID: {process.pid})")
    print("[INFO] Сервис доступен на http://127.0.0.1:5003/invocations")

if __name__ == "__main__":
    deploy_model()