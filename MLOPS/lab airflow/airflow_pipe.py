import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, OrdinalEncoder, PowerTransformer
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.linear_model import SGDRegressor
from sklearn.metrics import root_mean_squared_error
import numpy as np
import matplotlib.pyplot as plt
from datetime import datetime
from airflow import DAG
from airflow.operators.python import PythonOperator
from pathlib import Path
import os
from datetime import timedelta
from train_model import train

def prepare_data():
    """Подготовка данных для обучения"""
    # Загружаем локальный файл
    df = pd.read_csv("./students.csv")
    
    print(f"Исходный датасет: {df.shape}")
    print(f"Колонки: {df.columns.tolist()}")
    
    # Анализ пропущенных значений
    print(f"Пропущенные значения:\n{df.isnull().sum()}")
    
    # Удаляем student_id и week, так как это идентификаторы
    df = df.drop(columns=['student_id', 'week'], errors='ignore')
    
    # Анализ статистик
    print(f"Статистика по данным:\n{df.describe()}")
    
    # Проверка на бесконечные значения
    df = df.replace([np.inf, -np.inf], np.nan)
    
    # Удаляем строки с пропущенными значениями
    df = df.dropna()
    
    # Анализ выбросов для целевой переменной (performance_index)
    Q1 = df['performance_index'].quantile(0.25)
    Q3 = df['performance_index'].quantile(0.75)
    IQR = Q3 - Q1
    lower_bound = Q1 - 1.5 * IQR
    upper_bound = Q3 + 1.5 * IQR
    
    outliers = df[(df['performance_index'] < lower_bound) | 
                  (df['performance_index'] > upper_bound)]
    print(f"Найдено выбросов в performance_index: {len(outliers)}")
    
    # Можно либо удалить выбросы, либо оставить
    # df = df[(df['performance_index'] >= lower_bound) & 
    #         (df['performance_index'] <= upper_bound)]
    
    # Анализ корреляций с целевой переменной
    correlations = df.corr()['performance_index'].sort_values(ascending=False)
    print(f"Корреляции с performance_index:\n{correlations}")
    
    # Сохраняем очищенные данные
    df.to_csv('./students_clear.csv', index=False)
    print(f"Очищенный датасет: {df.shape}")
    
    return True

def analyze_features():
    """Анализ важности признаков"""
    df = pd.read_csv("./students_clear.csv")
    
    # Визуализация распределения целевой переменной
    plt.figure(figsize=(10, 6))
    plt.hist(df['performance_index'], bins=50, alpha=0.7, color='blue')
    plt.title('Распределение performance_index')
    plt.xlabel('performance_index')
    plt.ylabel('Частота')
    plt.savefig('./performance_distribution.png')
    plt.close()
    
    # Матрица корреляций
    plt.figure(figsize=(12, 10))
    corr_matrix = df.corr()
    plt.imshow(corr_matrix, cmap='coolwarm', aspect='auto')
    plt.colorbar()
    plt.xticks(range(len(corr_matrix.columns)), corr_matrix.columns, rotation=90)
    plt.yticks(range(len(corr_matrix.columns)), corr_matrix.columns)
    plt.title('Матрица корреляций')
    plt.tight_layout()
    plt.savefig('./correlation_matrix.png')
    plt.close()
    
    # Корреляция с целевой переменной
    correlations = df.corr()['performance_index'].drop('performance_index').sort_values(ascending=False)
    
    plt.figure(figsize=(10, 6))
    correlations.plot(kind='bar', color='green', alpha=0.7)
    plt.title('Корреляция признаков с performance_index')
    plt.xlabel('Признаки')
    plt.ylabel('Корреляция')
    plt.xticks(rotation=45, ha='right')
    plt.tight_layout()
    plt.savefig('./feature_correlations.png')
    plt.close()
    
    print("Анализ признаков завершен, графики сохранены")
    return True

# Определение DAG
dag_students = DAG(
    dag_id="train_students_pipe",
    start_date=datetime(2025, 2, 3),
    concurrency=4,
    schedule_interval=timedelta(minutes=5),
    max_active_runs=1,
    catchup=False,
    description="Pipeline for training model on students dataset"
)

# Определение задач
prepare_task = PythonOperator(
    python_callable=prepare_data, 
    task_id="prepare_students_data", 
    dag=dag_students
)

analyze_task = PythonOperator(
    python_callable=analyze_features, 
    task_id="analyze_students_features", 
    dag=dag_students
)

train_task = PythonOperator(
    python_callable=train, 
    task_id="train_students_model", 
    dag=dag_students
)

# Определение порядка выполнения
prepare_task >> analyze_task >> train_task