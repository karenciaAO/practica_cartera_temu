
# API Riesgo TEMU

API REST (FastAPI) para scoring de clientes (probabilidad de pérdida de cartera) y evaluación offline del modelo de riesgo TEMU.  
Incluye endpoints `/score` para un solo cliente, `/score-batch` para lotes y `/evaluate` para evaluar datasets completos.
## Requisitos

- Python 3.11
- fastapi
- uvicorn[standard]
- python-multipart
- pandas==2.0.3
- numpy==1.26.4
- scikit-learn==1.3.0
- joblib==1.3.2
- pyxlsb
- openpyxl
## Instalación

```bash
conda create -n temuapi python=3.11 -y
conda activate temuapi
pip install -r requirements.txt

o con `pip install …` directamente.

---

## 4️⃣ Arrancar la API  

```markdown
## Arrancar el servidor

Desde la carpeta donde está `app.py`:

```bash
python -m uvicorn app:app --reload
Por defecto se lanza en http://127.0.0.1:8000


---

## 5️⃣ Endpoints  

```markdown
## Endpoints

- **GET /health** → OK si el servicio está arriba.
- **GET /version** → Info del modelo cargado.
- **POST /score** → Recibe un JSON con un cliente y devuelve `p_perdida`.
- **POST /score-batch** → Recibe CSV/XLSX/XLSB o lista JSON y devuelve CSV de probabilidades.
- **POST /evaluate** → Evalúa un dataset completo con corte temporal y métricas.
Ejemplo de payload para /score:
{
  "IdentificadorCliente": 55027,
  "FechaEvento": "2023-10-18T18:14:19.630Z",
  "FechaVinculacionCliente": 44713.5,
  "FechaPrimerUso": 44727.5,
  "FechaUltimoUso": 45199.5,
  "UsabilidadCupo": 0.201138,
  "DiasMaximosMoraCreditosGenerados": 0,
  "NumeroCreditosGPrevius": 0,
  "NumeroCreditosGCanalFPrevius": 0,
  "NumeroCreditosGEstadoActivosPrevius": 0,
  "NumeroCreditosGEstadoPagadosPrevius": 0,
  "NumeroCreditosGCanalVPrevius": 0,
  "NumeroCreditosLPrevius": 0,
  "NumeroCreditosLEstadoActivosPrevius": 0,
  "NumeroCreditosLEstadoPagadosPrevius": 0,
  "TotalPagosEfectuadosGlobalmentePrevius": 0,
  "TotalPagosEfectuadosLocalmentePrevius": 0,
  "NumeroIntentosFallidos": 0,
  "CupoAprobado": 1000000,
  "ScoreCrediticio": 668,
  "Edad": 23,
  "CategoriaPrincipalCredito": "belleza-y-cuidado-personal",
  "UsoAppWeb": "App",
  "Genero": "Femenino",
  "TipoMunicipioEntregaTC": "INTERMEDIO",
  "CanalMunicipioEntregaTC": "Fisico",
  "CodigoAlmacenEntregaTC": 366898,
  "CodigoMunicipioEntregaTC": 2,
  "DiasMora": 0
}
## Vista Swagger UI

![Swagger UI](/Users/karenaraque/Desktop/Screenshot 2025-09-28 at 8.22.45 PM.png)
