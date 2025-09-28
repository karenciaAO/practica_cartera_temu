# train_validate.py
# ============================================================
# MODELO DE RIESGO — VALIDACIÓN TEMPORAL ESTRICTA (SIN FUGA)
# ============================================================

import os
import json
import numpy as np
import pandas as pd
from pathlib import Path
import warnings
warnings.filterwarnings("ignore")

from pandas.api.types import is_numeric_dtype

from sklearn.preprocessing import OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.calibration import CalibratedClassifierCV
from sklearn.utils.class_weight import compute_sample_weight
from sklearn.metrics import (
    roc_auc_score, average_precision_score, brier_score_loss,
    classification_report, confusion_matrix, precision_recall_curve
)

import joblib

# -----------------------------
# 0) CONFIG
# -----------------------------
TARGET = "PerdidaCartera"
RANDOM_STATE = 42
ARTIF_DIR = Path("./artifacts_modelo"); ARTIF_DIR.mkdir(parents=True, exist_ok=True)
REPORTS_DIR = Path("./artifacts_modelo/reports"); REPORTS_DIR.mkdir(parents=True, exist_ok=True)

np.set_printoptions(precision=3, suppress=True)
pd.set_option("display.max_rows", 200)
pd.set_option("display.max_columns", 200)
pd.set_option("display.width", 160)

# ============================================================
# 1) INGESTA — parquet o Excel .xlsb
#   - Puedes definir:
#       CLIENTES_PARQUET=/ruta/archivo.parquet
#       CLIENTES_XLSB=/ruta/archivo.xlsb
#   - Si no defines nada, buscará ./data/DataFramePrueba 2025_08.xlsb
# ============================================================
def load_clientes() -> pd.DataFrame:
    # 1) Intentar parquet (si está definido)
    pq_path = os.environ.get("CLIENTES_PARQUET", "").strip()
    if pq_path:
        print(f">>> Cargando clientes desde PARQUET: {pq_path}")
        df = pd.read_parquet(pq_path)
        return df

    # 2) Intentar XLSB por env var
    xlsb_env = os.environ.get("CLIENTES_XLSB", "").strip()
    if xlsb_env:
        print(f">>> Cargando clientes desde XLSB (env): {xlsb_env}")
        df = pd.read_excel(xlsb_env, engine="pyxlsb", sheet_name=0)
        return df

    # 3) Intentar XLSB por defecto en ./data
    default_xlsb = Path("./data") / "DataFramePrueba 2025_08.xlsb"
    if default_xlsb.exists():
        print(f">>> Cargando clientes desde XLSB (default): {default_xlsb}")
        df = pd.read_excel(default_xlsb, engine="pyxlsb", sheet_name=0)
        return df

    # 4) Si nada funcionó, intentar la ruta absoluta que nos pasaste (macOS)
    mac_abs = "/Users/karenaraque/Desktop/practica_cartera_temu/DataFramePrueba 2025_08.xlsb"
    if Path(mac_abs).exists():
        print(f">>> Cargando clientes desde XLSB (ruta absoluta): {mac_abs}")
        df = pd.read_excel(mac_abs, engine="pyxlsb", sheet_name=0)
        return df

    raise FileNotFoundError(
        "No se encontró la fuente de datos. "
        "Define CLIENTES_PARQUET o CLIENTES_XLSB, o coloca el XLSB en ./data/"
    )

clientes = load_clientes()

# Normalizar el target si viene como texto
if TARGET in clientes.columns and clientes[TARGET].dtype == object:
    print(">>> Normalizando target string → 0/1")
    clientes[TARGET] = (
        clientes[TARGET].astype(str).str.strip().str.lower()
        .map({"perdida":1, "no_perdida":0, "1":1, "0":0})
        .astype(int)
    )

assert TARGET in clientes.columns, f"No encuentro columna target '{TARGET}' en 'clientes'"
print(">>> SHAPE crudo:", clientes.shape)
print(">>> Columnas (primeras 20):", list(clientes.columns)[:20])

# ============================================================
# 2) FEATURE ENGINEERING — SIN FUGA
# ============================================================
def parse_and_features(df_raw: pd.DataFrame) -> pd.DataFrame:
    df = df_raw.copy()

    # --- Fechas
    df['FechaEvento_dt'] = (
        pd.to_datetime(df.get('FechaEvento'), errors='coerce', utc=True)
          .dt.tz_convert(None)
    )
    df['FechaVinculacionCliente_dt'] = pd.to_datetime(
        df.get('FechaVinculacionCliente'), errors='coerce',
        origin='1899-12-30', unit='D'
    )
    df['FechaUltimoUso_dt'] = pd.to_datetime(
        df.get('FechaUltimoUso'), errors='coerce',
        origin='1899-12-30', unit='D'
    )
    df['FechaPrimerUso_dt'] = pd.to_datetime(
        df.get('FechaPrimerUso'), errors='coerce',
        origin='1904-01-01', unit='D'
    )

    # Corrección conservadora PrimerUso
    df['Flag_PrimerUsoAntesVinc'] = (df['FechaPrimerUso_dt'] < df['FechaVinculacionCliente_dt']).astype(int)
    mask_bad = df['FechaPrimerUso_dt'].isna() | (df['FechaPrimerUso_dt'] < df['FechaVinculacionCliente_dt'])
    df.loc[mask_bad,  'FechaPrimerUso_corr'] = df.loc[mask_bad,  'FechaVinculacionCliente_dt']
    df.loc[~mask_bad, 'FechaPrimerUso_corr'] = df.loc[~mask_bad, 'FechaPrimerUso_dt']

    # Diferencias seguras
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

    # --- Usabilidad
    df['UsabilidadCupo'] = pd.to_numeric(df.get('UsabilidadCupo'), errors='coerce')
    df['Flag_Usab_NaN']    = df['UsabilidadCupo'].isna().astype(int)
    df['Flag_Usab_Outlier']= ((df['UsabilidadCupo'] < 0) | (df['UsabilidadCupo'] > 2)).astype(int)
    df.loc[(df['UsabilidadCupo'] < 0) | (df['UsabilidadCupo'] > 2), 'UsabilidadCupo'] = np.nan
    df['UsabilidadCupo']   = df['UsabilidadCupo'].fillna(df['UsabilidadCupo'].median())

    # --- Flags de fechas
    df['Flag_UltimoUsoPosterior'] = (df['FechaUltimoUso_dt'] > df['FechaEvento_dt']).fillna(False).astype(int)

    # --- Numéricas con nulos —> flags + imputación conservadora
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

    # --- Score & Cupo
    df['ScoreSinInfo'] = (df.get('ScoreCrediticio', 0) == 0).astype(int)
    if 'ScoreCrediticio' in df.columns and df['ScoreCrediticio'].isna().any():
        med_pos = df.loc[df['ScoreCrediticio']>0, 'ScoreCrediticio'].median()
        df['ScoreCrediticio'] = df['ScoreCrediticio'].fillna(med_pos)

    df['log_CupoAprobado'] = np.log1p(df.get('CupoAprobado'))
    df['log_CupoAprobado'] = df['log_CupoAprobado'].fillna(df['log_CupoAprobado'].median())

    # --- Categóricas limpias
    for c in ['CategoriaPrincipalCredito','UsoAppWeb','Genero','TipoMunicipioEntregaTC','CanalMunicipioEntregaTC']:
        if c in df.columns:
            df[c] = df[c].fillna('Desconocido')
    if 'Genero' in df.columns:
        df['Genero'] = df['Genero'].replace({27:'Desconocido'})
    if 'TipoMunicipioEntregaTC' in df.columns:
        df['TipoMunicipioEntregaTC'] = df['TipoMunicipioEntregaTC'].replace({'PEQUEÃ‘O':'PEQUEÑO'}).fillna('Desconocido')

    # --- Rango de Edad
    if 'Edad' in df.columns:
        df['Flag_Edad_Out'] = (~df['Edad'].between(18, 100, inclusive='both')).fillna(False).astype(int)
        df.loc[df['Flag_Edad_Out']==1, 'Edad'] = np.nan
        df['Edad'] = df['Edad'].fillna(df['Edad'].median())
    else:
        df['Flag_Edad_Out'] = 0
        df['Edad'] = 0

    # --- Features derivadas clave
    if 'Flag_PrimerUsoTemu' not in df.columns and 'NumeroCreditosGPrevius' in df.columns:
        df['Flag_PrimerUsoTemu'] = (df['NumeroCreditosGPrevius'] == 0).astype(int)

    df['ratio_pagos_local_global'] = (
        df.get('TotalPagosEfectuadosLocalmentePrevius', 0)
        / (df.get('TotalPagosEfectuadosGlobalmentePrevius', 0) + 1.0)
    )
    df['creditos_activos_ratio'] = (
        df.get('NumeroCreditosGEstadoActivosPrevius', 0)
        / (df.get('NumeroCreditosGPrevius', 0) + 1.0)
    )
    for c in ['ratio_pagos_local_global','creditos_activos_ratio']:
        df[c] = pd.to_numeric(df[c], errors='coerce').replace([np.inf,-np.inf], np.nan).fillna(0).clip(0,1)

    df['Flag_CanalVirtual'] = (
        (df.get('CanalMunicipioEntregaTC', '').astype(str).str.lower()=='virtual') |
        (df.get('TipoMunicipioEntregaTC', '').astype(str).str.upper()=='VIRTUAL')
    ).astype(int)

    # --- Score negativo -> 0, más bucket
    if 'ScoreCrediticio' in df.columns:
        df['Flag_Score_Negativo'] = (df['ScoreCrediticio'] < 0).astype(int)
        df.loc[df['ScoreCrediticio'] < 0, 'ScoreCrediticio'] = 0
    else:
        df['Flag_Score_Negativo'] = 0
        df['ScoreCrediticio'] = 0

    df['ScoreBucket'] = 'sin_info'
    mask_pos = df['ScoreCrediticio'] > 0
    if mask_pos.sum() > 0:
        b1, b2 = df.loc[mask_pos, 'ScoreCrediticio'].quantile([0.33, 0.66]).values
        df.loc[mask_pos & (df['ScoreCrediticio'] <= b1), 'ScoreBucket'] = 'bajo'
        df.loc[mask_pos & (df['ScoreCrediticio'] >  b1) & (df['ScoreCrediticio'] <= b2), 'ScoreBucket'] = 'medio'
        df.loc[mask_pos & (df['ScoreCrediticio'] >  b2), 'ScoreBucket'] = 'alto'
    df['ScoreBucket'] = pd.Categorical(df['ScoreBucket'], categories=['sin_info','bajo','medio','alto'], ordered=True)

    # --- Winsorización p99 + flags (colas largas)
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
            p99 = pd.to_numeric(df[c], errors='coerce').fillna(0).quantile(0.99)
            capped, flag = cap_with_flag(pd.to_numeric(df[c], errors='coerce').fillna(0), p99)
            df[c] = capped
            df[f'Flag_{c}_Capped'] = flag

    # Agrupar categorías raras de CategoriaPrincipalCredito (<0.1%)
    if 'CategoriaPrincipalCredito' in df.columns:
        vc = df['CategoriaPrincipalCredito'].astype(str).value_counts(dropna=False)
        cutoff = df.shape[0] * 0.001
        rare_levels = vc[vc < cutoff].index
        df['CategoriaPrincipalCredito'] = df['CategoriaPrincipalCredito'].astype(str)
        df.loc[df['CategoriaPrincipalCredito'].isin(rare_levels), 'CategoriaPrincipalCredito'] = 'OtrosRare'

    return df


def build_feature_lists(df: pd.DataFrame):
    drop_from_features = [
        'IdentificadorCliente','FechaEvento','FechaVinculacionCliente','FechaPrimerUso','FechaUltimoUso',
        'FechaEvento_dt','FechaVinculacionCliente_dt','FechaPrimerUso_dt','FechaUltimoUso_dt','FechaPrimerUso_corr',
        'CodigoAlmacenEntregaTC','CodigoAlmacenEntregaTC_str','AlmacenTop20',
        'CodigoMunicipioEntregaTC','MunicipioCat','MunicipioTop20','MesCompra',
        'DiasMora'
    ]
    num_base = [
        'UsabilidadCupo','MesesDesdeVinculacion','MesesDesdePrimerUso','DiasDesdeUltimoUso',
        'NumeroCreditosGPrevius','NumeroCreditosGCanalFPrevius','NumeroCreditosGCanalVPrevius',
        'NumeroCreditosGEstadoActivosPrevius','NumeroCreditosGEstadoPagadosPrevius',
        'NumeroCreditosLPrevius','NumeroCreditosLEstadoActivosPrevius','NumeroCreditosLEstadoPagadosPrevius',
        'TotalPagosEfectuadosGlobalmentePrevius','TotalPagosEfectuadosLocalmentePrevius',
        'NumeroIntentosFallidos','ScoreCrediticio','ScoreSinInfo','CupoAprobado','log_CupoAprobado','Edad',
        'Flag_Usab_NaN','Flag_Usab_Outlier','Flag_PrimerUsoAntesVinc','Flag_UltimoUsoPosterior','Flag_Edad_Out',
        'ratio_pagos_local_global','creditos_activos_ratio','Flag_Score_Negativo'
    ]
    num_dyn = [c for c in df.columns if c.startswith('Flag_') and (c.endswith('_NaN') or c.endswith('_Capped'))]
    num_final = [c for c in (num_base + num_dyn) if c in df.columns and c not in drop_from_features]

    cat_final = [
        'Genero','TipoMunicipioEntregaTC','CanalMunicipioEntregaTC','UsoAppWeb','CategoriaPrincipalCredito',
        'Flag_PrimerUsoTemu','ScoreBucket'
    ]
    cat_final = [c for c in cat_final if c in df.columns and c not in drop_from_features]

    # Evitar solape entre num y cat
    overlap = set(num_final) & set(cat_final)
    if overlap:
        print(f">>> Aviso: {len(overlap)} columnas estaban en num y cat. Se quitan de num: {sorted(overlap)}")
        num_final = [c for c in num_final if c not in overlap]

    # Unicidad y orden estable
    num_final = list(dict.fromkeys(num_final))
    cat_final = list(dict.fromkeys(cat_final))
    return num_final, cat_final


def prune_low_variance(X: pd.DataFrame):
    low_var = [c for c in X.columns if X[c].nunique(dropna=False) <= 1]
    if low_var:
        print(">>> Poda de columnas casi-constantes:", low_var)
        return X.drop(columns=low_var, errors='ignore'), low_var
    else:
        print(">>> Poda de columnas casi-constantes: ninguna")
        return X, []


def build_pipelines(cat_cols, num_cols):
    pre_logit = ColumnTransformer(
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

    logit = Pipeline(steps=[
        ('pre', pre_logit),
        ('clf', LogisticRegression(
            solver='saga', penalty='l2', class_weight='balanced',
            max_iter=800, n_jobs=-1, random_state=RANDOM_STATE
        ))
    ])

    pre_hgb = ColumnTransformer(
        transformers=[
            ('cat', OneHotEncoder(handle_unknown='ignore', sparse=False), cat_cols),
            ('num', 'passthrough', num_cols),
        ],
        remainder='drop'
    )

    hgb = Pipeline(steps=[
        ('pre', pre_hgb),
        ('clf', HistGradientBoostingClassifier(
            learning_rate=0.07, max_leaf_nodes=31, l2_regularization=1.0,
            random_state=RANDOM_STATE,
            early_stopping=True, validation_fraction=0.1, n_iter_no_change=20
        ))
    ])
    return logit, hgb


def evaluate_proba(y_true, y_proba, name="model"):
    auc = roc_auc_score(y_true, y_proba)
    ap  = average_precision_score(y_true, y_proba)
    br  = brier_score_loss(y_true, y_proba)
    print(f"[{name}] ROC-AUC={auc:.3f} | PR-AUC={ap:.3f} | Brier={br:.3f}")
    print(f"[{name}] Base rate y=1: {y_true.mean():.3f} | mean(p1)={np.mean(y_proba):.3f} | mean(p0)={np.mean(1-y_proba):.3f}")
    p, r, thr = precision_recall_curve(y_true, y_proba)
    f1 = 2*p*r/(p+r+1e-12)
    j  = np.argmax(f1)
    best_thr = thr[j-1] if j>0 and j-1 < len(thr) else 0.5
    y_hat = (y_proba >= best_thr).astype(int)
    print(f"[{name}] best-F1={f1[j]:.3f} @ thr={best_thr:.3f}")
    print(f"[{name}] ConfMatrix @thr={best_thr:.3f}:\n", confusion_matrix(y_true, y_hat))
    print(f"[{name}] Report @thr={best_thr:.3f}:\n", classification_report(y_true, y_hat, digits=3))
    return dict(roc_auc=auc, pr_auc=ap, brier=br, thr=best_thr, pr=p, rc=r, thr_curve=thr)


def find_threshold_by_target(p, r, thr, kind="precision", target=0.60):
    if kind == "precision":
        idx = np.where(p >= target)[0]
        if len(idx) == 0:
            return None
        k = idx[np.argmax(r[idx])]
        return float(thr[k-1]) if k > 0 else 0.5
    elif kind == "recall":
        idx = np.where(r >= target)[0]
        if len(idx) == 0:
            return None
        k = idx[np.argmax(p[idx])]
        return float(thr[k-1]) if k > 0 else 0.5
    else:
        return None


def operating_points(y_true, y_proba, name="model"):
    p, r, thr = precision_recall_curve(y_true, y_proba)
    f1 = 2*p*r/(p+r+1e-12)
    pts = {}
    j = np.argmax(f1); pts['maxF1'] = (thr[j-1] if j>0 else 0.5, p[j], r[j], f1[j])
    idx = np.where(p>=0.6)[0]
    if len(idx)>0:
        k = idx[np.argmax(r[idx])]
        pts['prec>=0.6'] = (thr[k-1] if k>0 else 0.5, p[k], r[k], f1[k])
    idx = np.where(r>=0.7)[0]
    if len(idx)>0:
        k = idx[np.argmax(p[idx])]
        pts['rec>=0.7'] = (thr[k-1] if k>0 else 0.5, p[k], r[k], f1[k])

    print(f"\n[{name}] Puntos de operación sugeridos:")
    for kk,(t,pp,rr,ff) in pts.items():
        print(f" - {kk:10s}: thr={t:.3f} | precision={pp:.3f} | recall={rr:.3f} | F1={ff:.3f}")
    return pts


def walk_forward_cv(df, X, y, cat_cols, num_cols, cut_fracs=(0.6,0.7,0.8)):
    cutoffs = df['FechaEvento_dt'].quantile(list(cut_fracs)).values
    rows_l, rows_h = [], []
    logit, hgb = build_pipelines(cat_cols, num_cols)

    print("\n>>> Walk-forward CV (sin fuga):")
    for i, c in enumerate(cutoffs, 1):
        tr_idx = df['FechaEvento_dt'] <= c
        va_idx = df['FechaEvento_dt'] >  c
        X_tr, X_va = X.loc[tr_idx], X.loc[va_idx]
        y_tr, y_va = y.loc[tr_idx], y.loc[va_idx]
        print(f"  - Split {i}: train={X_tr.shape}, valid={X_va.shape}, cutoff={pd.Timestamp(c).date()}")

        # Logit + calibración
        logit.fit(X_tr, y_tr)
        cal_logit = CalibratedClassifierCV(logit, method='sigmoid', cv='prefit')
        cal_logit.fit(X_va, y_va)
        proba_l = cal_logit.predict_proba(X_va)[:,1]
        m_l = evaluate_proba(y_va, proba_l, name=f"Logit+Cal (WF{i})")
        rows_l.append({"cutoff": c, **{k:v for k,v in m_l.items() if k in ['roc_auc','pr_auc','brier','thr']}})

        # HGB + sample_weight + calibración
        sw = compute_sample_weight("balanced", y_tr)
        hgb.fit(X_tr, y_tr, clf__sample_weight=sw)
        cal_hgb = CalibratedClassifierCV(hgb, method='isotonic', cv='prefit')
        cal_hgb.fit(X_va, y_va)
        proba_h = cal_hgb.predict_proba(X_va)[:,1]
        m_h = evaluate_proba(y_va, proba_h, name=f"HGB+Cal (WF{i})")
        rows_h.append({"cutoff": c, **{k:v for k,v in m_h.items() if k in ['roc_auc','pr_auc','brier','thr']}})

    return pd.DataFrame(rows_l), pd.DataFrame(rows_h)


# ============================================================
# 3) CONSTRUIR DATASET LIMPIO + LISTAS DE FEATURES + EDA
# ============================================================
print("\n>>> Construyendo features (sin fuga) ...")
clientes = parse_and_features(clientes)
num_final, cat_final = build_feature_lists(clientes)

X = clientes[num_final + cat_final].copy()
y = clientes[TARGET].astype(int).copy()

# 3A) EDA previa
print("\n===================== EDA PREVIA (SIN FUGA) =====================")
print(f"Shape de X: {X.shape} | y rate (1)= {y.mean():.3f}")
print("Rango de fechas (FechaEvento_dt):",
      str(clientes['FechaEvento_dt'].min().date()), "→", str(clientes['FechaEvento_dt'].max().date()))

vc = y.value_counts().rename({0:'no_perdida', 1:'perdida'})
print("\n[Balance de clases]")
print(pd.concat([vc, (vc/vc.sum()).round(3).rename('pct')], axis=1).to_string())

summary_rows = []
for c in X.columns:
    s = X[c]
    dtype = s.dtype
    pct_null = s.isnull().mean()*100
    nuni = s.nunique(dropna=False)
    top5 = s.value_counts(dropna=False).head(5).to_dict() if (not is_numeric_dtype(s)) else {}
    summary_rows.append({
        'columna': c, 'dtype': str(dtype), '%nulos': round(pct_null,1),
        'n_unicos': int(nuni), 'top5_valores': top5
    })
summary_df = pd.DataFrame(summary_rows).sort_values(['dtype','columna'])
print("\n[Resumen por columna] (primeras 30 filas)")
print(summary_df.head(30).to_string(index=False))
summary_df.to_csv(REPORTS_DIR/"eda_resumen_columnas.csv", index=False)

num_cols_for_corr = [c for c in X.columns if is_numeric_dtype(X[c]) and c != TARGET]
if len(num_cols_for_corr) > 0:
    corrs = X[num_cols_for_corr].corrwith(y).sort_values(ascending=False)
    print("\n[Correlación numéricas vs target] (top 20)")
    print(corrs.head(20).round(3).to_string())
    print("\n[Correlación numéricas vs target] (bottom 20)")
    print(corrs.tail(20).round(3).to_string())

cat_cols_for_xtab = [c for c in X.columns if not is_numeric_dtype(X[c])]
for c in cat_cols_for_xtab:
    if X[c].nunique(dropna=False) <= 25:
        tab = pd.crosstab(X[c], y, normalize='index').rename(columns={0:'no_perdida',1:'perdida'}).round(3)
        print(f"\n[Distribución {c} → proporción de pérdida por categoría]")
        print(tab.sort_values('perdida', ascending=False).to_string())

print("\n================= FIN EDA PREVIA (SIGUE MODELADO) =================")

# 3B) Limpiezas extra
dups_mask = X.columns.duplicated(keep='first')
if dups_mask.any():
    dups = pd.Series(X.columns)[dups_mask].tolist()
    print(f">>> Columnas duplicadas detectadas y removidas ({len(dups)}): {dups}")
    X = X.loc[:, ~X.columns.duplicated(keep='first')]
else:
    print(">>> Sin columnas duplicadas en X.")

X, dropped_lowvar = prune_low_variance(X)

print("\n=== Resumen columnas finales (previas al modelado) ===")
print(f"Numéricas: {len([c for c in X.columns if c in num_final])} | Categóricas: {len([c for c in X.columns if c in cat_final])}")
print("X shape:", X.shape, "| y rate (1):", y.mean().round(3))
print("FechaEvento_dt rango:", str(clientes['FechaEvento_dt'].min().date()), "→", str(clientes['FechaEvento_dt'].max().date()))

# ============================================================
# 4) WALK-FORWARD CV — SIN MIRAR EL FUTURO
# ============================================================
cat_cols = [c for c in cat_final if c in X.columns]
num_cols = [c for c in X.columns if c not in cat_cols]

logit_cv, hgb_cv = walk_forward_cv(clientes, X, y, cat_cols, num_cols)

print("\nLogit — Walk-forward CV:")
print(logit_cv)
print(f"Logit — medias: ROC-AUC={logit_cv.roc_auc.mean():.3f} | PR-AUC={logit_cv.pr_auc.mean():.3f} | Brier={logit_cv.brier.mean():.3f}")

print("\nHGB — Walk-forward CV:")
print(hgb_cv)
print(f"HGB — medias:   ROC-AUC={hgb_cv.roc_auc.mean():.3f} | PR-AUC={hgb_cv.pr_auc.mean():.3f} | Brier={hgb_cv.brier.mean():.3f}")

chosen = "HGB" if hgb_cv.pr_auc.mean() >= logit_cv.pr_auc.mean() else "Logit"
print(f"\n>>> Modelo elegido por PR-AUC medio: {chosen}")

# ============================================================
# 5) HOLDOUT FINAL (FUTURO) — 20% MÁS RECIENTE + CHEQUEOS
# ============================================================
cutoff = clientes['FechaEvento_dt'].quantile(0.8)
train_idx = clientes['FechaEvento_dt'] <= cutoff
hold_idx  = clientes['FechaEvento_dt'] >  cutoff
X_tr, X_ho = X.loc[train_idx], X.loc[hold_idx]
y_tr, y_ho = y.loc[train_idx], y.loc[hold_idx]
print(f"\n>>> Holdout temporal: train={X_tr.shape}, holdout={X_ho.shape}, corte={pd.Timestamp(cutoff).date()}")

idx_train = set(X_tr.index); idx_hold = set(X_ho.index)
assert idx_train.isdisjoint(idx_hold), "Solapamiento entre train y holdout"
assert clientes.loc[X_tr.index, 'FechaEvento_dt'].max() <= cutoff, "Train tiene fechas > cutoff"
assert clientes.loc[X_ho.index, 'FechaEvento_dt'].min() >  cutoff, "Holdout tiene fechas <= cutoff"
print(">>> Checks anti-fuga OK: sin solapes y split temporal correcto.")

logit, hgb = build_pipelines(cat_cols, num_cols)

# LOGIT + calibración
logit.fit(X_tr, y_tr)
cal_logit = CalibratedClassifierCV(logit, method='sigmoid', cv='prefit')
cal_logit.fit(X_ho, y_ho)
proba_logit = cal_logit.predict_proba(X_ho)[:,1]
_ = evaluate_proba(y_ho, proba_logit, name="HOLDOUT-Logit")
operating_points(y_ho, proba_logit, name="HOLDOUT-Logit")

# HGB + sample_weight + calibración
sw = compute_sample_weight("balanced", y_tr)
hgb.fit(X_tr, y_tr, clf__sample_weight=sw)
cal_hgb = CalibratedClassifierCV(hgb, method='isotonic', cv='prefit')
cal_hgb.fit(X_ho, y_ho)
proba_hgb = cal_hgb.predict_proba(X_ho)[:,1]
m_hold = evaluate_proba(y_ho, proba_hgb, name="HOLDOUT-HGB")
pts = operating_points(y_ho, proba_hgb, name="HOLDOUT-HGB")

# Barrido de umbrales para reporte
def sweep_thresholds(y_true, y_proba, thresholds=(0.20, 0.25, 0.265, 0.30, 0.40, 0.50)):
    rows = []
    for t in thresholds:
        yhat = (y_proba >= t).astype(int)
        TN, FP, FN, TP = confusion_matrix(y_true, yhat).ravel()
        prec1 = TP / (TP+FP) if TP+FP>0 else 0.0
        rec1  = TP / (TP+FN) if TP+FN>0 else 0.0
        f1    = (2*prec1*rec1/(prec1+rec1)) if (prec1+rec1)>0 else 0.0
        prec0 = TN / (TN+FN) if TN+FN>0 else 0.0
        rec0  = TN / (TN+FP) if TN+FP>0 else 0.0
        rows.append({
            "thr": t,
            "prec_1": round(prec1,3), "rec_1": round(rec1,3), "F1_1": round(f1,3),
            "prec_0": round(prec0,3), "rec_0": round(rec0,3),
            "TP": TP, "FP": FP, "FN": FN, "TN": TN
        })
    tab = pd.DataFrame(rows)
    print("\n[HOLDOUT-HGB] Barrido de umbrales (métricas por clase y confusión):")
    print(tab.to_string(index=False))
    tab.to_csv(REPORTS_DIR/"holdout_umbral_sweep.csv", index=False)
    return tab

tab_thr = sweep_thresholds(y_ho, proba_hgb)

# Punto de F1 máximo
p, r, thr = precision_recall_curve(y_ho, proba_hgb)
f1 = 2*p*r/(p+r+1e-12)
j  = np.argmax(f1)
thr_star = thr[j-1] if j>0 and j-1<len(thr) else 0.5
print(f"\n[Punto auto] Máxima F1 clase 1: thr≈{thr_star:.3f} | precision={p[j]:.3f} | recall={r[j]:.3f} | F1={f1[j]:.3f}")

# Calibración por deciles
def decile_calibration(y_true, y_proba, name="holdout"):
    bins = pd.qcut(y_proba, q=10, duplicates='drop')
    tab = pd.DataFrame({"p":y_proba, "y":y_true}).groupby(bins, observed=True).agg(
        p_mean=('p','mean'),
        y_rate=('y','mean'),
        n=('p','size')
    ).reset_index().rename(columns={"p":"bin"})
    print(f"\n[{name}] Calibración por deciles (p_mean vs y_rate):")
    print(tab.to_string(index=False))
    tab.to_csv(REPORTS_DIR/f"calibracion_deciles_{name}.csv", index=False)
    return tab

_ = decile_calibration(y_ho, proba_hgb, name="HOLDOUT-HGB")

# Umbrales por objetivo de negocio (opcional)
def find_threshold_by_target_simple(p, r, thr, kind="precision", target=0.60):
    return find_threshold_by_target(p, r, thr, kind=kind, target=target)

thr_prec06 = find_threshold_by_target_simple(p, r, thr, kind="precision", target=0.60)
thr_rec07  = find_threshold_by_target_simple(p, r, thr, kind="recall",    target=0.70)
print("\n[Umbrales negocio sugeridos]")
print(f" - precision>=0.60 -> thr≈{(thr_prec06 if thr_prec06 else float('nan')):.3f}")
print(f" - recall>=0.70    -> thr≈{(thr_rec07  if thr_rec07  else float('nan')):.3f}")
print(f" - maxF1           -> thr≈{thr_star:.3f}")

# Guardar bundle de artefactos (modelo calibrado + metadatos + umbrales)
bundle = {
    "model_calibrado": cal_hgb if (hgb_cv.pr_auc.mean() >= logit_cv.pr_auc.mean()) else cal_logit,
    "chosen": "HGB" if (hgb_cv.pr_auc.mean() >= logit_cv.pr_auc.mean()) else "Logit",
    "num_final": num_final,
    "cat_final": cat_final,
    "columns_after_prune": list(X.columns),
    "cutoff": cutoff,
    "thresholds": {
        "thr_maxF1": float(thr_star),
        "thr_prec06": float(thr_prec06) if thr_prec06 is not None else None,
        "thr_rec07": float(thr_rec07) if thr_rec07 is not None else None
    },
    "metrics_holdout": {
        "roc_auc": float(m_hold["roc_auc"]),
        "pr_auc":  float(m_hold["pr_auc"]),
        "brier":   float(m_hold["brier"])
    }
}
joblib.dump(bundle, ARTIF_DIR/"modelo_calibrado.joblib")
with open(ARTIF_DIR/"modelo_meta.json", "w") as f:
    json.dump({k:(v if not hasattr(v,'item') else v.item()) for k,v in bundle["metrics_holdout"].items()}, f, indent=2)

print(f"\n>>> Artefactos guardados en {ARTIF_DIR/'modelo_calibrado.joblib'}")
print(f">>> Meta y métricas en {ARTIF_DIR/'modelo_meta.json'}")
print(">>> Listo. Puedes pasar a levantar la API con api.py")
