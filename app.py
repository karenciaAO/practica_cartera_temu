# app.py
# ============================================================
# API Riesgo TEMU — FastAPI (score online + batch + evaluación offline)
# Requiere: fastapi, uvicorn, python-multipart, scikit-learn, pandas, numpy, joblib, pyxlsb, openpyxl
# ============================================================

from __future__ import annotations
import os, io, time, logging, traceback
from typing import List, Optional, Literal, Dict, Any

import numpy as np
import pandas as pd

from fastapi import FastAPI, File, UploadFile, HTTPException, Body, status
from fastapi.responses import JSONResponse, StreamingResponse, PlainTextResponse

from pydantic import BaseModel, Field

import joblib

# ---- scikit-learn
from sklearn.preprocessing import OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.calibration import CalibratedClassifierCV
from sklearn.metrics import (
    roc_auc_score, average_precision_score, brier_score_loss,
    confusion_matrix, precision_recall_curve
)
from sklearn.utils.class_weight import compute_sample_weight

# ---------------------------
# Config
# ---------------------------
APP_NAME = "API Riesgo TEMU"
APP_VERSION = "1.0.0"

MODEL_BUNDLE_PATH = os.getenv("MODEL_BUNDLE_PATH", "./artifacts_modelo/modelo_calibrado.joblib")
GOAL_PRECISION = float(os.getenv("GOAL_PRECISION", "0.80"))
RANDOM_STATE = int(os.getenv("RANDOM_STATE", "42"))

# Costeo para Expected Value en evaluación offline
COST_FP   = float(os.getenv("COST_FP", "1.0"))
COST_FN   = float(os.getenv("COST_FN", "5.0"))
BENEFIT_TP= float(os.getenv("BENEFIT_TP","4.0"))
COST_TN   = float(os.getenv("COST_TN", "0.0"))

TARGET = "PerdidaCartera"
CAT_CANDIDATES = ['CategoriaPrincipalCredito','UsoAppWeb','Genero',
                  'TipoMunicipioEntregaTC','CanalMunicipioEntregaTC']
EXCLUDE_FROM_FEATURES = {
    'IdentificadorCliente','DiasMora',
    'FechaEvento','FechaVinculacionCliente','FechaUltimoUso','FechaPrimerUso',
    'CodigoAlmacenEntregaTC','CodigoMunicipioEntregaTC'
}

# ---------------------------
# Logging
# ---------------------------
logger = logging.getLogger(APP_NAME)
handler = logging.StreamHandler()
handler.setFormatter(logging.Formatter("[%(asctime)s] [%(levelname)s] %(message)s"))
logger.addHandler(handler)
logger.setLevel(logging.INFO)

def timeit(fn):
    def _wrap(*args, **kwargs):
        t0 = time.time()
        try:
            return fn(*args, **kwargs)
        finally:
            logger.info(f"{fn.__name__} took {(time.time()-t0)*1000:.1f} ms")
    return _wrap

# ---------------------------
# FECHAS + FEATURES (idéntico a tu notebook resumido)
# ---------------------------
# ---------------------------
# FECHAS + FEATURES (robusto a ISO o Excel serial; tolera null)
# ---------------------------
def _to_datetime_iso_or_excel(s, excel_origin="1899-12-30"):
    """
    Intenta parsear primero como serial de Excel (numérico),
    y si no, como ISO/fecha libre. Devuelve una serie datetime (NaT si no se puede).
    """
    s_num = pd.to_numeric(s, errors='coerce')
    dt_excel = pd.to_datetime(s_num, errors='coerce', origin=excel_origin, unit='D')
    dt_iso   = pd.to_datetime(s, errors='coerce')  # ISO / string libre
    # preferimos Excel si hay número; si no, usamos ISO
    return dt_excel.combine_first(dt_iso)

def parse_dates_raw(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()

    # FechaEvento: normalmente viene como ISO string; admitimos ambos
    out['FechaEvento_dt'] = pd.to_datetime(out.get('FechaEvento'), errors='coerce', utc=True).dt.tz_convert(None)

    # Vinculación / Último uso: o serial Excel o ISO
    out['FechaVinculacionCliente_dt'] = _to_datetime_iso_or_excel(out.get('FechaVinculacionCliente'))
    out['FechaUltimoUso_dt_raw']      = _to_datetime_iso_or_excel(out.get('FechaUltimoUso'))

    # Primer uso: algunos exports usan epoch 1899 y otros 1904 → probamos ambos
    s = out.get('FechaPrimerUso')
    # ruta Excel 1899
    fpu_1899 = _to_datetime_iso_or_excel(s, excel_origin="1899-12-30")
    # ruta Excel 1904 (solo tiene sentido si había número)
    s_num = pd.to_numeric(s, errors='coerce')
    fpu_1904 = pd.to_datetime(s_num, errors='coerce', origin='1904-01-01', unit='D')
    # también intentamos ISO
    fpu_iso = pd.to_datetime(s, errors='coerce')

    # elegimos en orden: 1899 si válida, si no 1904, si no ISO
    tmp = fpu_1899.copy()
    mask_bad = tmp.isna() | (tmp.dt.year < 1900)
    tmp.loc[mask_bad] = fpu_1904.loc[mask_bad]
    mask_bad2 = tmp.isna() | (tmp.dt.year < 1900)
    tmp.loc[mask_bad2] = fpu_iso.loc[mask_bad2]

    out['FechaPrimerUso_dt_raw'] = tmp

    return out


def parse_and_features(
    df_raw: pd.DataFrame,
    primeruso_strategy: Literal["clip_a_vinc","poner_na","mantener"] = "clip_a_vinc",
    ultimo_uso_clip_evento: bool = True
) -> pd.DataFrame:
    df = df_raw.copy()
    df['Flag_PrimerUsoAntesVinc'] = (df['FechaPrimerUso_dt_raw'] < df['FechaVinculacionCliente_dt']).astype(int)
    if primeruso_strategy == "clip_a_vinc":
        df['FechaPrimerUso_dt'] = df['FechaPrimerUso_dt_raw']
        mask_bad = df['FechaPrimerUso_dt_raw'].isna() | (df['FechaPrimerUso_dt_raw'] < df['FechaVinculacionCliente_dt'])
        df.loc[mask_bad, 'FechaPrimerUso_dt'] = df.loc[mask_bad, 'FechaVinculacionCliente_dt']
    elif primeruso_strategy == "poner_na":
        df['FechaPrimerUso_dt'] = df['FechaPrimerUso_dt_raw']
        mask_bad = df['FechaPrimerUso_dt_raw'] < df['FechaVinculacionCliente_dt']
        df.loc[mask_bad, 'FechaPrimerUso_dt'] = pd.NaT
    elif primeruso_strategy == "mantener":
        df['FechaPrimerUso_dt'] = df['FechaPrimerUso_dt_raw']
    else:
        raise ValueError("primeruso_strategy inválida")

    df['Flag_UltimoUsoPosteriorEvento'] = (df['FechaUltimoUso_dt_raw'] > df['FechaEvento_dt']).astype(int)
    if ultimo_uso_clip_evento:
        df['FechaUltimoUso_dt'] = df['FechaUltimoUso_dt_raw'].copy()
        mask_bad2 = df['FechaUltimoUso_dt_raw'] > df['FechaEvento_dt']
        df.loc[mask_bad2, 'FechaUltimoUso_dt'] = df.loc[mask_bad2, 'FechaEvento_dt']
    else:
        df['FechaUltimoUso_dt'] = df['FechaUltimoUso_dt_raw']

    def safe_months(a, b):
        d = (a - b).dt.days
        d = d.where(d.notna(), 0); d = np.where(d < 0, 0, d)
        return d / 30.0
    def safe_days(a, b):
        d = (a - b).dt.days
        d = d.where(d.notna(), 0); d = np.where(d < 0, 0, d)
        return d

    df['MesesDesdeVinculacion'] = safe_months(df['FechaEvento_dt'], df['FechaVinculacionCliente_dt'])
    df['MesesDesdePrimerUso']   = safe_months(df['FechaEvento_dt'], df['FechaPrimerUso_dt'])
    df['DiasDesdeUltimoUso']    = safe_days(df['FechaEvento_dt'],  df['FechaUltimoUso_dt'])

    keep_nums = [
        'UsabilidadCupo','DiasMaximosMoraCreditosGenerados','NumeroCreditosGPrevius',
        'NumeroCreditosGCanalFPrevius','NumeroCreditosGEstadoActivosPrevius','NumeroCreditosGEstadoPagadosPrevius',
        'NumeroCreditosGCanalVPrevius','NumeroCreditosLPrevius','NumeroCreditosLEstadoActivosPrevius',
        'NumeroCreditosLEstadoPagadosPrevius','TotalPagosEfectuadosGlobalmentePrevius','TotalPagosEfectuadosLocalmentePrevius',
        'NumeroIntentosFallidos','CupoAprobado','ScoreCrediticio','Edad',
        'MesesDesdeVinculacion','MesesDesdePrimerUso','DiasDesdeUltimoUso'
    ]
    for c in keep_nums:
        if c in df.columns:
            df[f'Flag_{c}_NaN'] = df[c].isna().astype(int)

    df['UsabilidadCupo'] = pd.to_numeric(df.get('UsabilidadCupo', np.nan), errors='coerce')
    df['Flag_Usab_Outlier']= ((df['UsabilidadCupo'] < 0) | (df['UsabilidadCupo'] > 2)).astype(int)
    df.loc[(df['UsabilidadCupo'] < 0) | (df['UsabilidadCupo'] > 2), 'UsabilidadCupo'] = np.nan
    df['UsabilidadCupo'] = df['UsabilidadCupo'].fillna(df['UsabilidadCupo'].median())

    df['ScoreSinInfo'] = (df['ScoreCrediticio'].fillna(0) == 0).astype(int)
    df['ScoreCrediticio'] = df['ScoreCrediticio'].fillna(df['ScoreCrediticio'][df['ScoreCrediticio']>0].median())
    df.loc[df['ScoreCrediticio'] < 0, 'ScoreCrediticio'] = 0
    df['log_CupoAprobado'] = np.log1p(df['CupoAprobado'].fillna(df['CupoAprobado'].median()))

    for c in CAT_CANDIDATES:
        if c in df.columns:
            df[c] = df[c].fillna('Desconocido')
    if 'Genero' in df.columns:
        df['Genero'] = df['Genero'].replace({27:'Desconocido'})
    if 'TipoMunicipioEntregaTC' in df.columns:
        df['TipoMunicipioEntregaTC'] = df['TipoMunicipioEntregaTC'].replace({'PEQUEÃ‘O':'PEQUEÑO'}).fillna('Desconocido')

    df['Flag_PrimerUsoTemu'] = (df['NumeroCreditosGPrevius'].fillna(0) == 0).astype(int)
    df['ratio_pagos_local_global'] = (
        df['TotalPagosEfectuadosLocalmentePrevius'].fillna(0) /
        (df['TotalPagosEfectuadosGlobalmentePrevius'].fillna(0) + 1.0)
    ).clip(0,1)
    df['creditos_activos_ratio'] = (
        df['NumeroCreditosGEstadoActivosPrevius'].fillna(0) /
        (df['NumeroCreditosGPrevius'].fillna(0) + 1.0)
    ).clip(0,1)

    df['ScoreBucket'] = 'sin_info'
    mask_pos = df['ScoreCrediticio'] > 0
    if mask_pos.sum() > 0:
        q1, q2 = df.loc[mask_pos, 'ScoreCrediticio'].quantile([0.33, 0.66]).values
        df.loc[mask_pos & (df['ScoreCrediticio'] <= q1), 'ScoreBucket'] = 'bajo'
        df.loc[mask_pos & (df['ScoreCrediticio'] >  q1) & (df['ScoreCrediticio'] <= q2), 'ScoreBucket'] = 'medio'
        df.loc[mask_pos & (df['ScoreCrediticio'] >  q2), 'ScoreBucket'] = 'alto'
    df['ScoreBucket'] = pd.Categorical(df['ScoreBucket'], categories=['sin_info','bajo','medio','alto'], ordered=True)

    def cap_with_flag(s, upper):
        s = s.fillna(0)
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
            capped, flag = cap_with_flag(df[c], p99)
            df[c] = capped
            df[f'Flag_{c}_Capped'] = flag

    return df

def build_feature_lists(df: pd.DataFrame):
    drop = set(EXCLUDE_FROM_FEATURES) | {
        'FechaEvento_dt','FechaVinculacionCliente_dt','FechaUltimoUso_dt_raw','FechaPrimerUso_dt_raw'
    }
    num_base = [
        'UsabilidadCupo','MesesDesdeVinculacion','MesesDesdePrimerUso','DiasDesdeUltimoUso',
        'NumeroCreditosGPrevius','NumeroCreditosGCanalFPrevius','NumeroCreditosGEstadoActivosPrevius',
        'NumeroCreditosGEstadoPagadosPrevius','NumeroCreditosGCanalVPrevius',
        'NumeroCreditosLPrevius','NumeroCreditosLEstadoActivosPrevius','NumeroCreditosLEstadoPagadosPrevius',
        'TotalPagosEfectuadosGlobalmentePrevius','TotalPagosEfectuadosLocalmentePrevius',
        'NumeroIntentosFallidos','ScoreCrediticio','ScoreSinInfo','CupoAprobado','log_CupoAprobado','Edad',
        'Flag_PrimerUsoAntesVinc','Flag_Usab_Outlier','ratio_pagos_local_global','creditos_activos_ratio',
        'Flag_UltimoUsoPosteriorEvento'
    ]
    num_dyn = [c for c in df.columns if c.startswith('Flag_') and (c.endswith('_NaN') or c.endswith('_Capped'))]
    num_final = [c for c in (num_base + num_dyn) if c in df.columns and c not in drop]
    cat_final = [c for c in (CAT_CANDIDATES + ['Flag_PrimerUsoTemu','ScoreBucket']) if c in df.columns and c not in drop]
    overlap = set(num_final) & set(cat_final)
    if overlap:
        num_final = [c for c in num_final if c not in overlap]
    num_final = list(dict.fromkeys(num_final))
    cat_final = list(dict.fromkeys(cat_final))
    return num_final, cat_final

# ---------------------------
# Carga de bundle (modelo calibrado + columnas)
# ---------------------------
class Bundle:
    model: Optional[CalibratedClassifierCV] = None
    cols_ok: Optional[List[str]] = None
    experiment_best: Optional[Dict[str, Any]] = None

bundle = Bundle()

def load_bundle(path: str = MODEL_BUNDLE_PATH):
    if not os.path.exists(path):
        logger.warning(f"Bundle no encontrado en {path}. Endpoints de score devolverán 503.")
        return
    pk = joblib.load(path)
    bundle.model = pk["model_calibrado"]
    bundle.cols_ok = pk["columns_after_prune"]
    bundle.experiment_best = pk.get("experiment_best", None)
    logger.info(f"Bundle cargado desde {path}. Columnas esperadas={len(bundle.cols_ok)}")

load_bundle()

# ---------------------------
# Esquemas (Pydantic v2)
# ---------------------------
class ScoreRequest(BaseModel):
    FechaEvento: str | None = None
    FechaVinculacionCliente: float | None = None
    FechaPrimerUso: float | None = None
    FechaUltimoUso: float | None = None
    model_config = {"extra": "allow"}  # acepta extras

class ScoreResponse(BaseModel):
    p_perdida: float = Field(..., ge=0.0, le=1.0)
    version: str
    goal_precision: float

class EvalConfig(BaseModel):
    primeruso_strategy: Literal["clip_a_vinc","poner_na","mantener"] = "mantener"
    ultimo_uso_clip_evento: bool = True
    top_k_list: List[int] = Field(default_factory=lambda: [1,2,5,10,20])
    holdout_q: float = 0.20
    costs: Dict[str, float] = Field(
        default_factory=lambda: {"COST_FP":COST_FP, "COST_FN":COST_FN, "BENEFIT_TP":BENEFIT_TP, "COST_TN":COST_TN}
    )

# ---------------------------
# FastAPI app
# ---------------------------
app = FastAPI(title=APP_NAME, version=APP_VERSION)

@app.get("/health", response_class=PlainTextResponse)
def health():
    return "ok"

@app.get("/version")
def version():
    return {"app": APP_NAME, "version": APP_VERSION, "bundle_loaded": bundle.model is not None}

# ---------------------------
# SCORE (single JSON)
# ---------------------------
@timeit
@app.post("/score", response_model=ScoreResponse)
def score(payload: ScoreRequest):
    if bundle.model is None or bundle.cols_ok is None:
        raise HTTPException(status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
                            detail="Modelo no cargado. Sube el bundle o ajusta MODEL_BUNDLE_PATH.")
    try:
        row = pd.DataFrame([payload.model_dump(mode="python")])
        row = parse_dates_raw(row)

        primeruso = "clip_a_vinc"
        clip_ult = True
        if bundle.experiment_best:
            primeruso = bundle.experiment_best.get("primeruso", primeruso)
            clip_ult  = bool(bundle.experiment_best.get("clip_ultimo", clip_ult))

        row = parse_and_features(row, primeruso_strategy=primeruso, ultimo_uso_clip_evento=clip_ult)
        Xn = row.reindex(columns=bundle.cols_ok, fill_value=np.nan)

        proba1 = float(bundle.model.predict_proba(Xn)[:,1][0])
        return ScoreResponse(p_perdida=proba1, version=APP_VERSION, goal_precision=GOAL_PRECISION)
    except Exception as e:
        logger.error(traceback.format_exc())
        raise HTTPException(status_code=400, detail=f"Error procesando score: {e}")

# ---------------------------
# SCORE BATCH (archivo CSV/XLSX/XLSB o JSON list)
# ---------------------------
@timeit
@app.post("/score-batch")
def score_batch(file: UploadFile = File(None), rows: Optional[List[Dict[str, Any]]] = Body(None)):
    if bundle.model is None or bundle.cols_ok is None:
        raise HTTPException(status_code=503, detail="Modelo no cargado.")
    try:
        if file is None and rows is None:
            raise HTTPException(400, "Envía un archivo (CSV/XLSX/XLSB) o un JSON array.")

        if rows is not None:
            df = pd.DataFrame(rows)
        else:
            content = upload_read(file)
            df = load_table_from_bytes(content, file.filename)

        df = parse_dates_raw(df)

        primeruso = "clip_a_vinc"
        clip_ult = True
        if bundle.experiment_best:
            primeruso = bundle.experiment_best.get("primeruso", primeruso)
            clip_ult  = bool(bundle.experiment_best.get("clip_ultimo", clip_ult))

        df = parse_and_features(df, primeruso_strategy=primeruso, ultimo_uso_clip_evento=clip_ult)
        Xn = df.reindex(columns=bundle.cols_ok, fill_value=np.nan)
        proba1 = bundle.model.predict_proba(Xn)[:,1]

        out = pd.DataFrame({"p_perdida": proba1})
        buf = io.StringIO(); out.to_csv(buf, index=False); buf.seek(0)
        return StreamingResponse(buf, media_type="text/csv",
                                 headers={"Content-Disposition":"attachment; filename=score_batch.csv"})
    except HTTPException:
        raise
    except Exception as e:
        logger.error(traceback.format_exc())
        raise HTTPException(400, f"Error en score-batch: {e}")

def upload_read(upload: UploadFile) -> bytes:
    return upload.file.read()

def load_table_from_bytes(content: bytes, filename: str) -> pd.DataFrame:
    name = (filename or "").lower()
    bio = io.BytesIO(content)
    if name.endswith(".csv"):
        return pd.read_csv(bio)
    if name.endswith(".xlsx"):
        return pd.read_excel(bio, engine="openpyxl")
    if name.endswith(".xlsb"):
        return pd.read_excel(bio, engine="pyxlsb")
    try:
        return pd.read_csv(io.BytesIO(content))
    except Exception:
        raise HTTPException(400, "Formato no soportado. Usa CSV/XLSX/XLSB.")

# ---------------------------
# EVALUACIÓN OFFLINE
# ---------------------------
@timeit
@app.post("/evaluate")
def evaluate(file: UploadFile = File(...), cfg: EvalConfig = Body(default_factory=EvalConfig)):
    try:
        content = upload_read(file)
        df = load_table_from_bytes(content, file.filename)

        df = parse_dates_raw(df)
        if df['FechaEvento_dt'].isna().all():
            raise HTTPException(400, "No pude parsear FechaEvento en el archivo.")

        df_feat = parse_and_features(df,
                                     primeruso_strategy=cfg.primeruso_strategy,
                                     ultimo_uso_clip_evento=cfg.ultimo_uso_clip_evento)
        if TARGET not in df_feat.columns:
            raise HTTPException(400, f"El dataset no tiene la columna target '{TARGET}'.")

        num_final, cat_final = build_feature_lists(df_feat)
        X = df_feat[num_final + cat_final].copy()
        y = df_feat[TARGET].astype(int).copy()

        cut1 = df_feat['FechaEvento_dt'].quantile(0.64)
        cut2 = df_feat['FechaEvento_dt'].quantile(0.80)
        train_in_idx = df_feat['FechaEvento_dt'] <= cut1
        valid_in_idx = (df_feat['FechaEvento_dt'] > cut1) & (df_feat['FechaEvento_dt'] <= cut2)
        holdout_idx  = df_feat['FechaEvento_dt'] >  cut2

        X_tr, y_tr = X.loc[train_in_idx], y.loc[train_in_idx]
        X_va, y_va = X.loc[valid_in_idx], y.loc[valid_in_idx]
        X_ho, y_ho = X.loc[holdout_idx],  y.loc[holdout_idx]

        pipe = build_model(cat_final, [c for c in X.columns if c not in cat_final])
        sw = compute_sample_weight("balanced", y_tr)
        pipe.fit(X_tr, y_tr, clf__sample_weight=sw)

        cal = CalibratedClassifierCV(pipe, method='isotonic', cv='prefit')
        cal.fit(X_va, y_va)

        proba_va = cal.predict_proba(X_va)[:,1]
        proba_ho = cal.predict_proba(X_ho)[:,1]

        def metrics_block(y_true, p1) -> Dict[str, Any]:
            auc = roc_auc_score(y_true, p1)
            ap  = average_precision_score(y_true, p1)
            br  = brier_score_loss(y_true, p1)
            p, r, thr = precision_recall_curve(y_true, p1)
            f1 = 2*p*r/(p+r+1e-12)
            j  = int(np.argmax(f1))
            best_thr = float(thr[j-1]) if j>0 and j-1 < len(thr) else 0.5
            yhat = (p1 >= best_thr).astype(int)
            TN, FP, FN, TP = confusion_matrix(y_true, yhat).ravel()
            return {
                "roc_auc": float(auc), "pr_auc": float(ap), "brier": float(br),
                "best_f1": float(f1[j]), "best_thr": float(best_thr),
                "confusion": {"TN":int(TN),"FP":int(FP),"FN":int(FN),"TP":int(TP)},
                "base_rate": float(np.mean(y_true)),
                "mean_p1": float(np.mean(p1)),
            }

        valid_metrics = metrics_block(y_va, proba_va)
        hold_metrics  = metrics_block(y_ho, proba_ho)

        def topk(y_true, p1, ks):
            n = len(y_true)
            order = np.argsort(-p1)
            y_sorted = np.array(y_true)[order]
            out = []
            base = y_true.mean()
            cum = np.cumsum(y_sorted)
            for k in ks:
                m = max(1, int(n*k/100.0))
                tp_k = int(cum[m-1])
                rate_k = tp_k/m
                lift = (rate_k/base) if base>0 else None
                out.append({"top_%":k, "n_alertas":m, "morosos_detectados":tp_k,
                            "tasa_moros_topk":round(rate_k,3),
                            "lift_vs_base": round(lift,2) if lift else None})
            return out

        lift_holdout = topk(y_ho, proba_ho, cfg.top_k_list)

        def expected_value(y_true, p1, thr, c_fp, c_fn, b_tp, c_tn):
            yhat = (p1 >= thr).astype(int)
            TN, FP, FN, TP = confusion_matrix(y_true, yhat).ravel()
            EV = TP*b_tp - FP*c_fp - FN*c_fn - TN*c_tn
            return EV, TP, FP, FN, TN

        grid_thr = np.unique(np.round(np.linspace(0.01, 0.99, 99), 3))
        ev_rows = []
        for t in grid_thr:
            ev, TP, FP, FN, TN = expected_value(y_ho, proba_ho, t,
                                                cfg.costs["COST_FP"], cfg.costs["COST_FN"],
                                                cfg.costs["BENEFIT_TP"], cfg.costs["COST_TN"])
            ev_rows.append({"thr":float(t),"EV":float(ev),"TP":int(TP),"FP":int(FP),"FN":int(FN),"TN":int(TN)})
        ev_best = sorted(ev_rows, key=lambda d: d["EV"], reverse=True)[0]

        result = {
            "splits": {
                "train_in": int(train_in_idx.sum()),
                "valid_in": int(valid_in_idx.sum()),
                "holdout":  int(holdout_idx.sum()),
                "cut1": str(pd.Timestamp(cut1)),
                "cut2": str(pd.Timestamp(cut2)),
            },
            "metrics": {"valid": valid_metrics, "holdout": hold_metrics},
            "lift_holdout_topk": lift_holdout,
            "best_economic_threshold": ev_best,
            "goal_precision": GOAL_PRECISION,
            "config_used": cfg.model_dump(),
        }
        return JSONResponse(result)
    except HTTPException:
        raise
    except Exception as e:
        logger.error(traceback.format_exc())
        raise HTTPException(400, f"Error en evaluate: {e}")

def build_model(cat_cols, num_cols):
    pre = ColumnTransformer(
        transformers=[
            ('cat', Pipeline(steps=[
                ('imp', SimpleImputer(strategy='most_frequent')),
                ('ohe', OneHotEncoder(handle_unknown='ignore', sparse=False))
            ]), cat_cols),
            ('num', Pipeline(steps=[
                ('imp', SimpleImputer(strategy='median'))
            ]), num_cols)
        ],
        remainder='drop'
    )
    clf = HistGradientBoostingClassifier(
        learning_rate=0.07, max_leaf_nodes=31, l2_regularization=1.0,
        random_state=RANDOM_STATE
    )
    pipe = Pipeline([('pre', pre), ('clf', clf)])
    return pipe
