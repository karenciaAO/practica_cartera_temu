# api.py
import uvicorn
import joblib
import numpy as np
import pandas as pd
from pathlib import Path
from fastapi import FastAPI
from pydantic import BaseModel, Field
from typing import List, Optional, Any, Dict
import json
import warnings
warnings.filterwarnings("ignore")

ARTIF_DIR = Path("./artifacts_modelo")
BUNDLE_PATH = ARTIF_DIR/"modelo_calibrado.joblib"
META_PATH   = ARTIF_DIR/"modelo_meta.json"

# --------- Reusar funciones críticas (idénticas al training) ----------
def parse_and_features(df_raw: pd.DataFrame) -> pd.DataFrame:
    df = df_raw.copy()

    df['FechaEvento_dt'] = (
        pd.to_datetime(df['FechaEvento'], errors='coerce', utc=True)
          .dt.tz_convert(None)
    )
    df['FechaVinculacionCliente_dt'] = pd.to_datetime(
        df['FechaVinculacionCliente'], errors='coerce',
        origin='1899-12-30', unit='D'
    )
    df['FechaUltimoUso_dt'] = pd.to_datetime(
        df['FechaUltimoUso'], errors='coerce',
        origin='1899-12-30', unit='D'
    )
    df['FechaPrimerUso_dt'] = pd.to_datetime(
        df['FechaPrimerUso'], errors='coerce',
        origin='1904-01-01', unit='D'
    )

    df['Flag_PrimerUsoAntesVinc'] = (df['FechaPrimerUso_dt'] < df['FechaVinculacionCliente_dt']).astype(int)
    mask_bad = df['FechaPrimerUso_dt'].isna() | (df['FechaPrimerUso_dt'] < df['FechaVinculacionCliente_dt'])
    df.loc[mask_bad,  'FechaPrimerUso_corr'] = df.loc[mask_bad,  'FechaVinculacionCliente_dt']
    df.loc[~mask_bad, 'FechaPrimerUso_corr'] = df.loc[~mask_bad, 'FechaPrimerUso_dt']

    def safe_months(a, b):
        d = (a - b).dt.days
        d = d.where(d.notna(), 0); d = np.where(d < 0, 0, d)
        return d / 30.0

    def safe_days(a, b):
        d = (a - b).dt.days
        d = d.where(d.notna(), 0); d = np.where(d < 0, 0, d)
        return d

    df['MesesDesdeVinculacion'] = safe_months(df['FechaEvento_dt'], df['FechaVinculacionCliente_dt'])
    df['MesesDesdePrimerUso']   = safe_months(df['FechaEvento_dt'], df['FechaPrimerUso_corr'])
    df['DiasDesdeUltimoUso']    = safe_days(df['FechaEvento_dt'],  df['FechaUltimoUso_dt'])

    df['UsabilidadCupo'] = pd.to_numeric(df['UsabilidadCupo'], errors='coerce')
    df['Flag_Usab_NaN']    = df['UsabilidadCupo'].isna().astype(int)
    df['Flag_Usab_Outlier']= ((df['UsabilidadCupo'] < 0) | (df['UsabilidadCupo'] > 2)).astype(int)
    df.loc[(df['UsabilidadCupo'] < 0) | (df['UsabilidadCupo'] > 2), 'UsabilidadCupo'] = np.nan
    df['UsabilidadCupo']   = df['UsabilidadCupo'].fillna(df['UsabilidadCupo'].median())

    df['Flag_UltimoUsoPosterior'] = (df['FechaUltimoUso_dt'] > df['FechaEvento_dt']).fillna(False).astype(int)

    num_cols_candidates = [
        'DiasMaximosMoraCreditosGenerados',
        'NumeroCreditosGPrevius','NumeroCreditosGCanalFPrevius','NumeroCreditosGCanalVPrevius',
        'NumeroCreditosGEstadoActivosPrevius','NumeroCreditosGEstadoPagadosPrevius',
        'NumeroCreditosLPrevius','NumeroCreditosLEstadoActivosPrevius','NumeroCreditosLEstadoPagadosPrevius',
        'TotalPagosEfectuadosGlobalmentePrevius','TotalPagosEfectuadosLocalmentePrevius',
        'NumeroIntentosFallidos','CupoAprobado','ScoreCrediticio','Edad',
        'MesesDesdeVinculacion','MesesDesdePrimerUso','DiasDesdeUltimoUso'
    ]
    for c in num_cols_candidates:
        if c in df.columns:
            df[f'Flag_{c}_NaN'] = df[c].isna().astype(int)
            if c.startswith('NumeroCreditos') or c.startswith('TotalPagos'):
                df[c] = df[c].fillna(0)
            elif c in ['DiasMaximosMoraCreditosGenerados','Edad','NumeroIntentosFallidos']:
                df[c] = df[c].fillna(0)

    df['ScoreSinInfo'] = (df['ScoreCrediticio'] == 0).astype(int)
    if df['ScoreCrediticio'].isna().any():
        med_pos = df.loc[df['ScoreCrediticio']>0, 'ScoreCrediticio'].median()
        df['ScoreCrediticio'] = df['ScoreCrediticio'].fillna(med_pos)

    df['log_CupoAprobado'] = np.log1p(df['CupoAprobado'])
    df['log_CupoAprobado'] = df['log_CupoAprobado'].fillna(df['log_CupoAprobado'].median())

    for c in ['CategoriaPrincipalCredito','UsoAppWeb','Genero','TipoMunicipioEntregaTC','CanalMunicipioEntregaTC']:
        if c in df.columns:
            df[c] = df[c].fillna('Desconocido')
    df['Genero'] = df['Genero'].replace({27:'Desconocido'})
    df['TipoMunicipioEntregaTC'] = df['TipoMunicipioEntregaTC'].replace({'PEQUEÃ‘O':'PEQUEÑO'}).fillna('Desconocido')

    df['Flag_Edad_Out'] = (~df['Edad'].between(18, 100, inclusive='both')).fillna(False).astype(int)
    df.loc[df['Flag_Edad_Out']==1, 'Edad'] = np.nan
    df['Edad'] = df['Edad'].fillna(df['Edad'].median())

    if 'Flag_PrimerUsoTemu' not in df.columns and 'NumeroCreditosGPrevius' in df.columns:
        df['Flag_PrimerUsoTemu'] = (df['NumeroCreditosGPrevius'] == 0).astype(int)

    df['ratio_pagos_local_global'] = (
        df['TotalPagosEfectuadosLocalmentePrevius'].fillna(0) /
        (df['TotalPagosEfectuadosGlobalmentePrevius'].fillna(0) + 1.0)
    )
    df['creditos_activos_ratio'] = (
        df['NumeroCreditosGEstadoActivosPrevius'].fillna(0) /
        (df['NumeroCreditosGPrevius'].fillna(0) + 1.0)
    )
    for c in ['ratio_pagos_local_global','creditos_activos_ratio']:
        df[c] = df[c].replace([np.inf,-np.inf], np.nan).fillna(0).clip(0,1)

    df['Flag_CanalVirtual'] = (
        (df['CanalMunicipioEntregaTC'].astype(str).str.lower()=='virtual') |
        (df['TipoMunicipioEntregaTC'].astype(str).str.upper()=='VIRTUAL')
    ).astype(int)

    df['Flag_Score_Negativo'] = (df['ScoreCrediticio'] < 0).astype(int)
    df.loc[df['ScoreCrediticio'] < 0, 'ScoreCrediticio'] = 0

    df['ScoreBucket'] = 'sin_info'
    mask_pos = df['ScoreCrediticio'] > 0
    if mask_pos.sum() > 0:
        b1, b2 = df.loc[mask_pos, 'ScoreCrediticio'].quantile([0.33, 0.66]).values
        df.loc[mask_pos & (df['ScoreCrediticio'] <= b1), 'ScoreBucket'] = 'bajo'
        df.loc[mask_pos & (df['ScoreCrediticio'] >  b1) & (df['ScoreCrediticio'] <= b2), 'ScoreBucket'] = 'medio'
        df.loc[mask_pos & (df['ScoreCrediticio'] >  b2), 'ScoreBucket'] = 'alto'
    df['ScoreBucket'] = pd.Categorical(df['ScoreBucket'], categories=['sin_info','bajo','medio','alto'], ordered=True)

    def cap_with_flag(s, upper):
        flag = (s > upper).astype(int)
        return np.where(s > upper, upper, s), flag

    cap_cols = [
        'DiasDesdeUltimoUso','MesesDesdeVinculacion',
        'TotalPagosEfectuadosGlobalmentePrevius','TotalPagosEfectuadosLocalmentePrevius',
        'NumeroCreditosGPrevius','NumeroCreditosGCanalFPrevius','NumeroCreditosGCanalVPrevius',
        'NumeroCreditosGEstadoActivosPrevius','NumeroCreditosGEstadoPagadosPrevius',
        'NumeroCreditosLPrevius','NumeroCreditosLEstadoActivosPrevius','NumeroCreditosLEstadoPagadosPrevius'
    ]
    for c in cap_cols:
        if c in df.columns:
            p99 = df[c].quantile(0.99)
            capped, flag = cap_with_flag(df[c].fillna(0), p99)
            df[c] = capped
            df[f'Flag_{c}_Capped'] = flag

    if 'CategoriaPrincipalCredito' in df.columns:
        vc = df['CategoriaPrincipalCredito'].astype(str).value_counts(dropna=False)
        cutoff = df.shape[0] * 0.001
        rare_levels = vc[vc < cutoff].index
        df['CategoriaPrincipalCredito'] = df['CategoriaPrincipalCredito'].astype(str)
        df.loc[df['CategoriaPrincipalCredito'].isin(rare_levels), 'CategoriaPrincipalCredito'] = 'OtrosRare'

    return df

# ------------------ Cargar artefacto -------------------------
print(f">>> Cargando bundle desde {BUNDLE_PATH} ...")
bundle = joblib.load(BUNDLE_PATH)
model  = bundle["model_calibrado"]
num_final = bundle["num_final"]; cat_final = bundle["cat_final"]
cols_ok = bundle["columns_after_prune"]
thresholds = bundle.get("thresholds", {})
thr_default = thresholds.get("thr_maxF1", 0.5)

meta = {}
if META_PATH.exists():
    with open(META_PATH) as f:
        meta = json.load(f)

# ------------------ FastAPI -------------------------
app = FastAPI(title="API Modelo Riesgo", version="1.0.0")

class Record(BaseModel):
    # Puedes incluir todas las columnas crudas que vengan en tu payload.
    # Pydantic permite extra=ignore por defecto: si falta alguna, el pipeline imputa.
    IdentificadorCliente: Optional[Any] = None
    FechaEvento: Optional[Any] = None
    UsabilidadCupo: Optional[float] = None
    CategoriaPrincipalCredito: Optional[str] = None
    DiasMaximosMoraCreditosGenerados: Optional[float] = None
    NumeroCreditosGPrevius: Optional[float] = None
    NumeroCreditosGCanalFPrevius: Optional[float] = None
    NumeroCreditosGEstadoActivosPrevius: Optional[float] = None
    NumeroCreditosGEstadoPagadosPrevius: Optional[float] = None
    NumeroCreditosGCanalVPrevius: Optional[float] = None
    NumeroCreditosLPrevius: Optional[float] = None
    NumeroCreditosLEstadoActivosPrevius: Optional[float] = None
    NumeroCreditosLEstadoPagadosPrevius: Optional[float] = None
    FechaVinculacionCliente: Optional[Any] = None
    FechaPrimerUso: Optional[Any] = None
    FechaUltimoUso: Optional[Any] = None
    TotalPagosEfectuadosGlobalmentePrevius: Optional[float] = None
    TotalPagosEfectuadosLocalmentePrevius: Optional[float] = None
    CodigoAlmacenEntregaTC: Optional[Any] = None
    CodigoMunicipioEntregaTC: Optional[Any] = None
    UsoAppWeb: Optional[str] = None
    Genero: Optional[str] = None
    TipoMunicipioEntregaTC: Optional[str] = None
    CanalMunicipioEntregaTC: Optional[str] = None
    NumeroIntentosFallidos: Optional[float] = None
    ScoreCrediticio: Optional[float] = None
    Edad: Optional[float] = None
    CupoAprobado: Optional[float] = None

class PredictRequest(BaseModel):
    records: List[Record] = Field(..., description="Lista de registros crudos (mismo esquema que training).")
    threshold: Optional[float] = Field(None, description="Umbral de decisión (si no, usa thr_maxF1 del bundle).")

@app.get("/health")
def health():
    return {"status": "ok", "model_loaded": True, "thr_default": thr_default}

@app.get("/meta")
def meta_endpoint():
    return {"metrics_holdout": meta, "thresholds": thresholds, "chosen": bundle.get("chosen")}

@app.post("/predict_proba")
def predict_proba(req: PredictRequest):
    df_raw = pd.DataFrame([r.dict() for r in req.records])
    df = parse_and_features(df_raw)
    Xn = df[num_final + cat_final].copy()
    Xn = Xn.reindex(columns=cols_ok, fill_value=np.nan)
    Xn = Xn.loc[:, ~Xn.columns.duplicated(keep='first')]
    p1 = model.predict_proba(Xn)[:,1]
    p0 = 1 - p1
    return {"probs": [{"p0": float(a), "p1": float(b)} for a,b in zip(p0, p1)]}

@app.post("/predict")
def predict(req: PredictRequest):
    thr = req.threshold if req.threshold is not None else thr_default
    df_raw = pd.DataFrame([r.dict() for r in req.records])
    df = parse_and_features(df_raw)
    Xn = df[num_final + cat_final].copy()
    Xn = Xn.reindex(columns=cols_ok, fill_value=np.nan)
    Xn = Xn.loc[:, ~Xn.columns.duplicated(keep='first')]
    p1 = model.predict_proba(Xn)[:,1]
    yhat = (p1 >= thr).astype(int)
    return {
        "threshold_used": thr,
        "preds": [{"yhat": int(int(y)), "p1": float(float(p)), "p0": float(1-float(p))} for y,p in zip(yhat, p1)]
    }

if __name__ == "__main__":
    # uvicorn api:app --reload
    uvicorn.run(app, host="0.0.0.0", port=8000)
