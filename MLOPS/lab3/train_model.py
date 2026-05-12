#!/usr/bin/env python3
"""
Этап 2: Обучение моделей с MLflow трекингом
"""
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, cross_val_score, KFold
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from sklearn.linear_model import LinearRegression, Ridge, Lasso
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor, HistGradientBoostingRegressor
import mlflow
import mlflow.sklearn
import warnings
warnings.filterwarnings('ignore')

def load_data():
    print("[INFO] Загрузка данных...")
    df = pd.read_csv('data_for_model.csv')
    
    feature_columns = ['Year', 'Mileage', 'Engine_size', 'Brand_id', 'Model_id', 'Fuel_type_id']
    X = df[feature_columns]
    y = df['Price']
    
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, shuffle=True
    )
    
    print(f"[INFO] Train: {X_train.shape}, Test: {X_test.shape}")
    return X_train, X_test, y_train, y_test

def train_models(X_train, X_test, y_train, y_test):
    print("[INFO] Настройка MLflow...")
    mlflow.set_experiment("Car_Price_Prediction")
    
    models = {
        "LinearRegression": LinearRegression(),
        "Ridge": Ridge(alpha=1.0),
        "Lasso": Lasso(alpha=0.1, max_iter=5000),
        "DecisionTree": DecisionTreeRegressor(random_state=42),
        "RandomForest": RandomForestRegressor(n_estimators=100, random_state=42, n_jobs=-1),
        "HistGradientBoosting": HistGradientBoostingRegressor(random_state=42, max_iter=300)
    }
    
    cv = KFold(n_splits=5, shuffle=True, random_state=42)
    
    print("\n[INFO] Обучение моделей с MLflow трекингом...")
    print("-" * 60)
    
    best_r2 = -1
    best_path = ""
    
    for name, model in models.items():
        with mlflow.start_run(run_name=name):
            # Логируем параметры
            if name == "RandomForest":
                mlflow.log_param("n_estimators", 100)
            elif name == "HistGradientBoosting":
                mlflow.log_param("max_iter", 300)
            elif name == "Ridge":
                mlflow.log_param("alpha", 1.0)
            elif name == "Lasso":
                mlflow.log_param("alpha", 0.1)
            mlflow.log_param("model_type", name)
            
            # Кросс-валидация
            scores = cross_val_score(model, X_train, y_train, cv=cv,
                                    scoring='neg_mean_squared_error', n_jobs=-1)
            rmse_scores = np.sqrt(-scores)
            cv_rmse_mean = rmse_scores.mean()
            cv_rmse_std = rmse_scores.std()
            
            mlflow.log_metric("cv_rmse_mean", cv_rmse_mean)
            mlflow.log_metric("cv_rmse_std", cv_rmse_std)
            
            # Обучение
            model.fit(X_train, y_train)
            y_pred = model.predict(X_test)
            
            # Метрики
            test_rmse = np.sqrt(mean_squared_error(y_test, y_pred))
            test_mae = mean_absolute_error(y_test, y_pred)
            test_r2 = r2_score(y_test, y_pred)
            
            mlflow.log_metric("test_rmse", test_rmse)
            mlflow.log_metric("test_mae", test_mae)
            mlflow.log_metric("test_r2", test_r2)
            
            # Сохраняем модель
            mlflow.sklearn.log_model(model, "model")
            
            print(f"{name:<25} | CV RMSE: {cv_rmse_mean:.4f} | Test RMSE: {test_rmse:.4f} | Test R²: {test_r2:.4f}")
            
            # Запоминаем лучшую модель
            if test_r2 > best_r2:
                best_r2 = test_r2
                run_id = mlflow.active_run().info.run_id
                best_path = f"runs:/{run_id}/model"
    
    print("-" * 60)
    
    # Выводим путь к лучшей модели (для deploy)
    with open("best_model.txt", "w") as f:
        f.write(best_path)
    
    print(f"\n[RESULT] Лучшая модель: {best_path} (R² = {best_r2:.4f})")
    
    return best_path

if __name__ == "__main__":
    X_train, X_test, y_train, y_test = load_data()
    train_models(X_train, X_test, y_train, y_test)