#!/usr/bin/env python3
"""
Этап 1: Получение датасета
"""
import os
import sys

LOCAL_DATA = "data_for_model.csv"

def download_data():
    print("[INFO] Проверка наличия датасета...")
    
    if not os.path.exists(LOCAL_DATA):
        print(f"[ERROR] Файл {LOCAL_DATA} не найден!")
        print("[INFO] Скачивание из GitHub...")
        import urllib.request
        url = "https://github.com/rafGleb/ml_lab1/blob/lab_3/MLOPS/lab3/data_for_model.csv"
        try:
            urllib.request.urlretrieve(url, LOCAL_DATA)
            print(f"[SUCCESS] Данные скачаны: {LOCAL_DATA}")
        except:
            print("[ERROR] Не удалось скачать данные")
            sys.exit(1)
    else:
        print(f"[INFO] Даиасет найден: {LOCAL_DATA}")
    
    size = os.path.getsize(LOCAL_DATA)
    print(f"[INFO] Размер файла: {size} байт")

if __name__ == "__main__":
    download_data()