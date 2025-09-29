# Informe Automático — Modelo de Riesgo Temu
_Validación temporal estricta (sin fuga)._

## 🧾 Resumen ejecutivo
- **ROC-AUC (holdout):** 0.829
- **PR-AUC (holdout):** 0.542
- **Brier (holdout):** 0.142
- **Variante ganadora:** `mantener__clip_ultimo` | PR-AUC(VALID)=0.757 | ROC-AUC(VALID)=0.871
- **Cortes temporales:** train≤**2023-05-13**, valid≤**2023-07-29**, holdout>**2023-07-29**

## 🔬 Comparativa de experimentos (VALID)
```
exp,primeruso,clip_ultimo,valid_pr_auc,valid_roc_auc,valid_brier,train_pr_auc,train_roc_auc
mantener__clip_ultimo,mantener,True,0.7565465557559835,0.8706122943891271,0.1171410347024606,0.7455627493142484,0.8840772566064999
clip_a_vinc__clip_ultimo,clip_a_vinc,True,0.7559222826669143,0.8700658781763201,0.1173972309790414,0.7473408549979492,0.8837771599203827
poner_na__clip_ultimo,poner_na,True,0.7553924245713564,0.8707294055547081,0.1170972012010996,0.7446467030149445,0.8835137664436282
```

## 📈 Métricas en HOLDOUT (final)
```
metric,value
ROC_AUC,0.8293882555463963
PR_AUC,0.5420351577338736
Brier,0.1418617828949929
```

## 🎯 Barrido de umbrales (clase=1)
```
thr,prec_1,rec_1,TP,FP,FN,TN
0.01,0.213,0.999,6093,22458,4,833
0.02,0.216,0.999,6089,22055,8,1236
0.03,0.231,0.994,6060,20148,37,3143
0.04,0.25,0.987,6019,18050,78,5241
0.05,0.25,0.987,6019,18049,78,5242
0.06,0.271,0.975,5946,15967,151,7324
0.07,0.272,0.974,5941,15874,156,7417
0.08,0.286,0.965,5883,14722,214,8569
0.09,0.311,0.943,5752,12761,345,10530
0.1,0.311,0.943,5752,12760,345,10531
0.11,0.311,0.943,5752,12760,345,10531
0.12,0.324,0.934,5695,11902,402,11389
0.13,0.342,0.915,5578,10741,519,12550
0.14,0.342,0.915,5578,10741,519,12550
0.15,0.342,0.914,5575,10720,522,12571
```

## 🚀 Ganancia/Lift por Top-k%
```
top_%,n_alertas,morosos_detectados,tasa_moros_topk,lift_vs_base
1,293,211,0.72,3.47
2,587,430,0.733,3.53
5,1469,1099,0.748,3.61
10,2938,1932,0.658,3.17
20,5877,3090,0.526,2.53
```

## 💰 Valor esperado por umbral
- **Mejor EV:** thr=0.14 | EV=8976.0 | TP=5578.0 FP=10741.0 FN=519.0 TN=12550.0

```
thr,EV,TP,FP,FN,TN
0.14,8976.0,5578,10741,519,12550
0.13,8976.0,5578,10741,519,12550
0.16,8970.0,5575,10720,522,12571
0.15,8970.0,5575,10720,522,12571
0.12,8868.0,5695,11902,402,11389
0.18,8846.0,5399,9260,698,14031
0.17,8844.0,5399,9262,698,14029
0.19,8738.0,5327,8720,770,14571
0.2,8738.0,5327,8720,770,14571
0.1,8523.0,5752,12760,345,10531
0.11,8523.0,5752,12760,345,10531
0.09,8522.0,5752,12761,345,10530
0.21,8434.0,5261,8430,836,14861
0.25,8387.0,5248,8360,849,14931
0.23,8386.0,5248,8361,849,14930
0.24,8386.0,5248,8361,849,14930
0.22,8386.0,5248,8361,849,14930
0.08,7740.0,5883,14722,214,8569
0.07,7110.0,5941,15874,156,7417
0.06,7062.0,5946,15967,151,7324
```

## ⚖️ Calibración por deciles
```
index,p_mean,y_rate,n
"(-0.001, 0.0277]",0.0183546223812048,0.0116352201257861,3180
"(0.0277, 0.0511]",0.0434700408861482,0.0265548567435359,4293
"(0.0511, 0.0873]",0.0836801012999584,0.0570252792475014,3402
"(0.0873, 0.118]",0.1168212442758556,0.0614035087719298,912
"(0.118, 0.169]",0.1471327995964302,0.1010547805375978,2939
"(0.169, 0.25]",0.2312605057741317,0.1692876965772433,3243
"(0.25, 0.351]",0.3027563762565058,0.2526676829268293,2624
"(0.351, 0.534]",0.4331707053108642,0.3798013245033113,3020
"(0.534, 0.83]",0.71272946047114,0.3929045092838196,3016
"(0.83, 1.0]",0.9291979837814188,0.6723450525552737,2759
```

## 🔍 Importancias por permutación (Top-20)
```
feature,importance_mean,importance_std
NumeroCreditosGEstadoActivosPrevius,0.1086467997046419,0.0025165047231102
NumeroCreditosLEstadoActivosPrevius,0.0774525195169424,0.003500875852693
NumeroCreditosGEstadoPagadosPrevius,0.0479526912734814,0.0013021179341805
creditos_activos_ratio,0.0478951733312581,0.0020115192346208
NumeroCreditosLEstadoPagadosPrevius,0.0370796611485256,0.001236277149681
UsabilidadCupo,0.0297891634127919,0.0014331451883692
ScoreCrediticio,0.0227045142362137,0.0009456729036983
DiasDesdeUltimoUso,0.0094877854562503,0.000500758324729
TotalPagosEfectuadosGlobalmentePrevius,0.009159974117312,0.0007187464752281
TotalPagosEfectuadosLocalmentePrevius,0.0069754946440013,0.0004192955022438
ScoreBucket,0.0059703537124678,0.0004475623007996
MesesDesdeVinculacion,0.0054726457848286,0.0004887606649516
TipoMunicipioEntregaTC,0.0039225252555643,0.0001761746423534
UsoAppWeb,0.0035590489371068,0.000172103935594
Edad,0.0015676123648594,0.0004606136916992
CategoriaPrincipalCredito,0.0015108545620179,0.0004135082857396
Flag_PrimerUsoTemu,0.0014410316990576,0.0001817775277497
Genero,0.0007387456091527,0.000150458729708
NumeroIntentosFallidos,0.0004008584487742,4.4826423776870096e-05
Flag_NumeroCreditosGPrevius_NaN,0.0003276858408489,0.0001164093017197
```

## 📝 Notas/lecturas rápidas
## 📌 Correlaciones y hallazgos rápidos

- **DiasMora** correlación alta con *PerdidaCartera* (~0.83). Es variable de **estado actual** → no usar como feature predictiva (evitar fuga).
- **Historial de créditos previos**: activos y pagados muestran señal (p. ej., `NumeroCreditosGEstadoActivosPrevius`, `NumeroCreditosLEstadoActivosPrevius`, `NumeroCreditosLPrevius`…).
- **Antigüedades** (meses desde vinculación / primer uso) aportan señal más estable que las fechas crudas.
- **Demografía y cupo** (`Edad`, `CupoAprobado`, `ScoreCrediticio`): correlación negativa moderada → más edad/cupo/score, menos pérdida.
- **Canal/Tipo municipio**: patrón “Virtual > Físico” en pérdida (≈28% vs 16–20%).
- **Género**: leve diferencia; mantener como categórica.

## ⚙️ Parámetros de costo/beneficio usados
```json
{
  "COST_FP": 1.0,
  "COST_FN": 5.0,
  "BENEFIT_TP": 4.0,
  "COST_TN": 0.0
}
```

