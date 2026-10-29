import pandas as pd
import numpy as np

from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression, ElasticNet
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error

import mlflow
from ucimlrepo import fetch_ucirepo
import mlflow.sklearn   

mlflow.set_experiment("simple_model")


def get_data():
    try:
        wine = fetch_ucirepo(id = 109)
        x = wine.data.features
        y = wine.data.targets
        
        return x, y
    except Exception as e:
        raise e

def evaluate_model(actual, predicted):
    mse = mean_squared_error(actual, predicted)
    mae = mean_absolute_error(actual, predicted)
    r2 = r2_score(actual, predicted)
    
    return mse, mae, r2

def train(alpha, l1_ratio):
    x, y = get_data()
    
    x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=21)
    
    with mlflow.start_run(run_name="ElasticNet_Final"):
        en = ElasticNet(alpha=alpha, l1_ratio=l1_ratio, random_state=21)
        en.fit(x_train, y_train)

        pred = en.predict(x_test)

        mse, mae, r2 = evaluate_model(y_test, pred)

        print(f"ElasticNet model parameters (alpha={alpha}, l1_ratio={l1_ratio}):")
        print(f"ElasticNet model performance: \n  MSE = {mse} \n  MAE = {mae} \n  R2 = {r2} ")

        params = {"alpha" : alpha, "l1_ratio" : l1_ratio}
        mlflow.log_params(params)

        metrics = {"mse" : mse, "mae" : mae, "r2":r2}
        mlflow.log_metrics(metrics)

        mlflow.sklearn.log_model(sk_model = en, name="model")

if __name__ == "__main__":
    alpha = 0.05
    l1_ratio = 0.05 
    train(alpha, l1_ratio)