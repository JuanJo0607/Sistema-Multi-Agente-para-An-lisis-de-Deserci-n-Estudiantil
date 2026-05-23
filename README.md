# Sistema Multi-Agente para Análisis de Deserción Estudiantil

**Proyecto Final — Inteligencia Artificial · Universidad EAFIT · 2026-1**

Sistema que combina machine learning clásico con un pipeline multi-agente (LangGraph + Groq + ChromaDB) para predecir la deserción estudiantil y generar reportes ejecutivos automáticos con contexto de literatura científica.

---

## Resultados del modelo

| Modelo | F1 (Dropout) | AUC-ROC |
|---|---|---|
| Dummy (baseline) | 0.00 | 0.50 |
| Logistic Regression | ~0.72 | ~0.87 |
| Random Forest ✓ | **0.7879** | **0.9335** |
| Gradient Boosting | ~0.78 | ~0.92 |

**RAG accuracy:** 10/10 (100%) en conjunto de evaluación de 10 preguntas.

---

## Requisitos

- Python 3.10 o superior
- API key gratuita de Groq: <https://console.groq.com/keys>
- Conexión a internet (primera ejecución descarga el dataset y el modelo de embeddings)

---

## Instalación

```bash
# 1. Clonar el repositorio
git clone <url-del-repositorio>
cd Sistema-Multi-Agente-para-An-lisis-de-Deserci-n-Estudiantil

# 2. Crear y activar entorno virtual
python -m venv .venv
source .venv/bin/activate        # Linux / Mac
# .venv\Scripts\activate         # Windows

# 3. Instalar dependencias
pip install -r requirements.txt

# 4. Configurar API key de Groq
cp .env.example .env
# Editar .env y reemplazar 'tu_api_key_aqui' con tu key real
```

---

## Ejecución

Los notebooks deben ejecutarse **en orden**. Cada uno genera archivos que usa el siguiente.

```bash
# Activar el entorno primero
source .venv/bin/activate

# Abrir Jupyter
jupyter notebook
```

| Orden | Notebook | Qué hace | Tiempo aprox. |
|---|---|---|---|
| 1 | `notebooks/01_eda.ipynb` | Descarga dataset, EDA, 5 visualizaciones | ~2 min |
| 2 | `notebooks/02_preprocessing.ipynb` | Split train/val/test sin data leakage | <1 min |
| 3 | `notebooks/03_modeling.ipynb` | Entrena 3 modelos, evalúa en test set | ~5 min |
| 4 | `notebooks/04_llm_rag_agents.ipynb` | RAG + 4 agentes LangGraph + reporte ejecutivo | ~3 min |

> **Nota:** El notebook 01 descarga el dataset de UCI (~500 KB) en la primera ejecución.
> El notebook 04 descarga el modelo de embeddings (~90 MB) en la primera ejecución.

---

## Parte gráfica

Los notebooks generan **10 figuras** guardadas en `docs/figures/`:

| Figura | Contenido |
|---|---|
| `01_distribucion_objetivo.png` | Distribución de clases (Dropout vs No Dropout) |
| `02_boxplots_outliers.png` | Variables con outliers comparadas por clase |
| `03_variables_academicas.png` | Distribución de variables del 1er semestre |
| `04_correlacion.png` | Matriz de correlación (top 14 variables) |
| `05_socioeconomicas.png` | Tasa de deserción por variables socioeconómicas |
| `06_tabla_modelos.png` | Tabla comparativa de 3 modelos ML |
| `07_curvas_aprendizaje.png` | Curvas de aprendizaje del mejor modelo |
| `08_evaluacion_final.png` | Matriz de confusión + Curva ROC + Curva PR |
| `09_feature_importance.png` | Top 15 variables más predictivas |
| `10_evaluacion_rag.png` | Accuracy del componente RAG por documento |

El reporte ejecutivo del sistema multi-agente se muestra en la salida del notebook 04.

### App interactiva (Streamlit)

Para una demo visual con interfaz web, ejecuta:

```bash
source .venv/bin/activate
streamlit run app/main.py
```

La app abre en `http://localhost:8501` y tiene 3 pestañas:
- **Sistema Multi-Agente** — ejecuta los 4 agentes con tu consulta y muestra el reporte ejecutivo
- **Evaluación del Modelo** — métricas y figuras del Random Forest
- **RAG — Literatura** — corpus de papers indexados y evaluación del retriever

---

## Arquitectura del sistema multi-agente

```
          ┌─── Agente EDA ───┐
START ────┤                  ├──→ Agente ML ──→ Agente Sintetizador ──→ Reporte
          └─── Agente RAG ───┘
               (ChromaDB)
```

- **Agente EDA:** interpreta estadísticos reales del dataset con el LLM
- **Agente RAG:** recupera contexto de 5 papers científicos vía ChromaDB
- **Agente ML:** interpreta métricas del modelo Random Forest guardado
- **Agente Sintetizador:** genera reporte ejecutivo integrando los 3 análisis

**LLM:** `llama-3.1-8b-instant` vía Groq (gratuito)
**Embeddings:** `all-MiniLM-L6-v2` vía HuggingFace (local, gratuito)

---

## Estructura del repositorio

```
├── README.md
├── requirements.txt
├── .env.example                        ← plantilla para configurar API key
├── corpus/                             ← 5 papers sobre deserción (para RAG)
├── data/
│   ├── raw/student_dropout.csv         ← dataset UCI (4,424 estudiantes)
│   └── processed/                      ← splits train/val/test
├── docs/
│   ├── figures/                        ← 10 figuras generadas
│   └── informe_final.pdf               ← informe LaTeX
├── models/checkpoints/
│   └── mejor_modelo.joblib             ← pipeline Random Forest entrenado
├── notebooks/
│   ├── 01_eda.ipynb
│   ├── 02_preprocessing.ipynb
│   ├── 03_modeling.ipynb
│   └── 04_llm_rag_agents.ipynb
└── src/agents/                         ← lógica de agentes LangGraph
```

---

## Video demo

[Ver demo one drive](https://eafit-my.sharepoint.com/:v:/g/personal/scadavidz_eafit_edu_co/IQAHsvdbaX8URpjeHLNTtYF3AXV9l5mgUY0njenZmLxmegk?e=Mq8DHK)

---

## Integrantes

- Juan José Restrepo Cardona — jjrestrepc@eafit.edu.co
- Samuel Cadavid Zapata — scadavidz@eafit.edu.co

---

## Dataset

Predict Students' Dropout and Academic Success  
Fuente principal (Kaggle): <https://www.kaggle.com/datasets/ankushpanday1/predict-students-dropout-and-academic-success>  
Fuente alternativa (UCI ML Repository id=697): <https://archive.ics.uci.edu/dataset/697/predict+students+dropout+and+academic+success>
