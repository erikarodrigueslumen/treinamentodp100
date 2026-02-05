import os
import argparse
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score
import joblib
import mlflow
import mlflow.sklearn

from utils import load_dataset, MODEL_DIR


def train(test_size: float = 0.2, random_state: int = 42, experiment: str = 'gelato-magico'):
    # MLflow – definir experimento
    mlflow.set_experiment(experiment)

    # Carregar dados
    df = load_dataset()
    X = df[['temperatura']]
    y = df['vendas']

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, random_state=random_state
    )

    with mlflow.start_run():
        model = LinearRegression()
        model.fit(X_train, y_train)

        preds = model.predict(X_test)
        mse = mean_squared_error(y_test, preds)
        r2 = r2_score(y_test, preds)

        # Logar parâmetros e métricas
        mlflow.log_param('test_size', test_size)
        mlflow.log_param('random_state', random_state)
        mlflow.log_metric('mse', mse)
        mlflow.log_metric('r2', r2)

        # Salvar modelo localmente (joblib) e via MLflow
        model_path = MODEL_DIR / 'modelo_sorvete.joblib'
        joblib.dump(model, model_path)
        mlflow.sklearn.log_model(model, artifact_path='modelo_sorvete')

        print(f"Treino concluído. MSE={mse:.3f} | R2={r2:.3f}")
        print(f"Modelo salvo em: {model_path}")


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Treinar modelo de vendas de sorvete')
    parser.add_argument('--test_size', type=float, default=0.2)
    parser.add_argument('--random_state', type=int, default=42)
    parser.add_argument('--experiment', type=str, default='gelato-magico')
    args = parser.parse_args()

    train(test_size=args.test_size, random_state=args.random_state, experiment=args.experiment)
