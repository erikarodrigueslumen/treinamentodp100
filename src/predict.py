import argparse
import numpy as np
import joblib
from utils import MODEL_DIR


def predict(temperatura: float) -> float:
    model_path = MODEL_DIR / 'modelo_sorvete.joblib'
    if not model_path.exists():
        raise FileNotFoundError(f"Modelo não encontrado em {model_path}. Treine primeiro com src/train.py")
    model = joblib.load(model_path)
    y_pred = model.predict(np.array([[temperatura]], dtype=float))
    return float(y_pred[0])


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Predizer vendas de sorvete a partir da temperatura (°C)')
    parser.add_argument('temperatura', type=float, help='Temperatura em °C')
    args = parser.parse_args()

    y = predict(args.temperatura)
    print(f"\U0001F4C8 Previsão de vendas: {y:.0f} unidades para {args.temperatura:.1f} °C")
