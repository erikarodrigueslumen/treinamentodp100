from fastapi import FastAPI, Query
import numpy as np
import joblib
from utils import MODEL_DIR

app = FastAPI(title='API – Gelato Mágico', description='Previsão de vendas de sorvete a partir da temperatura', version='1.0.0')

model_path = MODEL_DIR / 'modelo_sorvete.joblib'
model = None
if model_path.exists():
    model = joblib.load(model_path)


@app.get('/health')
def health():
    status = 'ok' if model is not None else 'model_not_loaded'
    return {'status': status}


@app.get('/predict')
def predict(temperatura: float = Query(..., description='Temperatura em °C')):
    if model is None:
        return {'error': 'Modelo não carregado. Treine com src/train.py e reinicie a API.'}
    y_pred = model.predict(np.array([[temperatura]], dtype=float))
    return {'temperatura': temperatura, 'vendas_previstas': float(y_pred[0])}
