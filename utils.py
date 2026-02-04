"""
Funções utilitárias para caminhos e carregamento de dados.
"""
from pathlib import Path
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / 'data'
MODEL_DIR = ROOT / 'model'
MODEL_DIR.mkdir(exist_ok=True)


def load_dataset(name: str = 'ice_cream_sales.csv') -> pd.DataFrame:
    """Carrega o dataset do diretório data/"""
    path = DATA / name
    if not path.exists():
        raise FileNotFoundError(f"Arquivo não encontrado: {path}")
    return pd.read_csv(path)

