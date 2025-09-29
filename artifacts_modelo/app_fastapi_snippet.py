
from fastapi import FastAPI
import joblib
import pandas as pd

app = FastAPI(title="API Riesgo Temu", version="1.0")

BUNDLE_PATH = "/Users/karenaraque/Desktop/practica_cartera_temu/artifacts_modelo/modelo_calibrado.joblib"

from __main__ import parse_and_features  # reutilizamos funciones

bundle = joblib.load(BUNDLE_PATH)
model = bundle["model_calibrado"]
cols_ok = bundle["columns_after_prune"]
best_exp = bundle.get("experiment_best", None)

@app.post("/score")
def score(payload: dict):
    df = pd.DataFrame([payload])
    primeruso = "clip_a_vinc" if not best_exp else best_exp.get("primeruso","clip_a_vinc")
    clip_ult  = True if not best_exp else bool(best_exp.get("clip_ultimo", True))
    df = parse_and_features(df, primeruso_strategy=primeruso, ultimo_uso_clip_evento=clip_ult)
    Xn = df.reindex(columns=cols_ok, fill_value={})
    proba1 = model.predict_proba(Xn)[:,1]
    return {"p_perdida": float(proba1[0])}
