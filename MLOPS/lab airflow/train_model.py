from os import name
from sklearn.preprocessing import StandardScaler, PowerTransformer
import pandas as pd
from sklearn.model_selection import train_test_split
import mlflow
from sklearn.linear_model import SGDRegressor
from sklearn.model_selection import GridSearchCV
import numpy as np
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from mlflow.models import infer_signature
import joblib


def scale_frame(frame):
    df = frame.copy()
    # Определяем целевую переменную (performance_index) и признаки
    target_col = 'performance_index'
    feature_cols = [col for col in df.columns if col not in ['student_id', 'week', target_col]]
    
    X = df[feature_cols]
    y = df[target_col]
    
    scaler = StandardScaler()
    power_trans = PowerTransformer()
    
    X_scale = scaler.fit_transform(X.values)
    Y_scale = power_trans.fit_transform(y.values.reshape(-1, 1))
    
    return X_scale, Y_scale, power_trans, scaler, feature_cols

def eval_metrics(actual, pred):
    rmse = np.sqrt(mean_squared_error(actual, pred))
    mae = mean_absolute_error(actual, pred)
    r2 = r2_score(actual, pred)
    return rmse, mae, r2


def train():
    # Загружаем локальный файл
    df = pd.read_csv("./students.csv")
    
    # Удаляем student_id и week, так как это идентификаторы
    df = df.drop(columns=['student_id', 'week'], errors='ignore')
    
    X, Y, power_trans, scaler, feature_cols = scale_frame(df)
    X_train, X_val, y_train, y_val = train_test_split(X, Y,
                                                    test_size=0.3,
                                                    random_state=42)
    
    params = {
        'alpha': [0.0001, 0.001, 0.01, 0.05, 0.1],
        'l1_ratio': [0.001, 0.05, 0.01, 0.2],
        "penalty": ["l1", "l2", "elasticnet"],
        "loss": ['squared_error', 'huber', 'epsilon_insensitive'],
        "fit_intercept": [False, True],
    }
    
    mlflow.set_experiment("linear_model_students")
    with mlflow.start_run():
        lr = SGDRegressor(random_state=42)
        clf = GridSearchCV(lr, params, cv=3, n_jobs=4)
        clf.fit(X_train, y_train.reshape(-1))
        best = clf.best_estimator_
        
        y_pred = best.predict(X_val)
        y_price_pred = power_trans.inverse_transform(y_pred.reshape(-1, 1))
        
        (rmse, mae, r2) = eval_metrics(
            power_trans.inverse_transform(y_val), 
            y_price_pred
        )
        
        # Логирование параметров
        mlflow.log_param("alpha", best.alpha)
        mlflow.log_param("l1_ratio", best.l1_ratio)
        mlflow.log_param("penalty", best.penalty)
        mlflow.log_param("eta0", best.eta0)
        mlflow.log_param("loss", best.loss)
        mlflow.log_param("fit_intercept", best.fit_intercept)
        mlflow.log_param("epsilon", best.epsilon)
        
        # Логирование метрик
        mlflow.log_metric("rmse", rmse)
        mlflow.log_metric("r2", r2)
        mlflow.log_metric("mae", mae)
        
        # Сохраняем модель и препроцессоры
        predictions = best.predict(X_train)
        signature = infer_signature(X_train, predictions)
        mlflow.sklearn.log_model(best, "model", signature=signature)
        
        # Сохраняем модели локально
        with open("sgd_students.pkl", "wb") as file:
            joblib.dump(best, file)
        
        # Сохраняем препроцессоры
        with open("scaler_students.pkl", "wb") as file:
            joblib.dump(scaler, file)
        
        with open("power_transformer_students.pkl", "wb") as file:
            joblib.dump(power_trans, file)
        
        # Сохраняем список признаков
        with open("feature_cols_students.txt", "w") as file:
            file.write("\n".join(feature_cols))
        
        print(f"Модель обучена. RMSE: {rmse:.4f}, R2: {r2:.4f}, MAE: {mae:.4f}")
        print(f"Лучшие параметры: {best.get_params()}")
    
    return best


if __name__ == "__main__":
    train()