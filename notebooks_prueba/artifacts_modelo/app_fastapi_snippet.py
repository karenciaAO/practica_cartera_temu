
from fastapi import FastAPI
import joblib
import pandas as pd

app = FastAPI(title="API Riesgo Temu", version="1.0")

BUNDLE_PATH = "/Users/karenaraque/Desktop/practica_cartera_temu/notebooks_prueba/artifacts_modelo/modelo_calibrado.joblib"

from __main__ import parse_and_features  # reutilizamos funciones de arriba

bundle = joblib.load(BUNDLE_PATH)
model = bundle["model_calibrado"]
cols_ok = bundle["columns_after_prune"]

@app.post("/score")
def score(payload: dict):
    df = pd.DataFrame([payload])
    df = parse_and_features(df)
    Xn = df.reindex(columns=cols_ok, fill_value={})
    proba1 = model.predict_proba(Xn)[:,1]
    return {"p_perdida": float(proba1[0])}
