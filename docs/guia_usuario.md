# Guía de usuario — Sistema Multi-Agente para Análisis de Deserción Estudiantil

---

## Antes de empezar

Necesitas tener instalado **Python 3.10 o superior**. Para saber tu versión escribe `python --version` en una terminal.

La API key para el LLM se incluye en el correo de entrega del proyecto. Una vez la tengas, crea un archivo llamado `.env` en la raíz del proyecto con este contenido:

```
GROQ_API_KEY=la_key_que_recibiste
```

---

## Instalación

Clona el repositorio y entra a la carpeta:

```bash
git clone git@github.com:JuanJo0607/Sistema-Multi-Agente-para-An-lisis-de-Deserci-n-Estudiantil.git
cd Sistema-Multi-Agente-para-An-lisis-de-Deserci-n-Estudiantil
```

Crea un entorno virtual para no mezclar las dependencias con las de otros proyectos:

```bash
python -m venv .venv
source .venv/bin/activate      # En Windows: .venv\Scripts\activate
```

Instala todo lo necesario. Este paso puede tardar unos minutos la primera vez porque descarga PyTorch y otros paquetes grandes:

```bash
pip install -r requirements.txt
```

---

## Cómo correr el proyecto

El proyecto tiene dos formas de usarse: con notebooks de Jupyter (el análisis completo paso a paso) o con la app interactiva de Streamlit la cual es más visual y rápida para demo.

### Opción 1 — Notebooks (análisis completo)

Abre Jupyter:

```bash
jupyter notebook
```

Se abre una pestaña en el navegador. Corre los notebooks **en este orden exacto**, porque cada uno genera archivos que usa el siguiente:

**Notebook 01 — EDA**
Descarga el dataset directamente de UCI, hace el análisis exploratorio y genera 5 gráficas. La primera vez que lo corras va a descargar el dataset, así que necesitas internet.

**Notebook 02 — Preprocesamiento**
Toma el dataset del notebook anterior, separa las variables y divide todo en conjuntos de entrenamiento, validación y test.

**Notebook 03 — Modelado**
Aquí es donde se entrenan los modelos. Compara tres algoritmos (Regresión Logística, Random Forest y Gradient Boosting) y evalúa el mejor en el conjunto de test.

**Notebook 04 — Agentes LLM + RAG**
Este es el corazón del proyecto. Configura el sistema multi-agente, construye la base de datos vectorial con los papers, y corre los 4 agentes para generar un reporte ejecutivo. La primera vez que lo corras descarga el modelo de embeddings, así que también necesitas internet.

### Opción 2 — App Streamlit 

Si solo quieres ver el sistema funcionando sin abrir notebooks, usa la app:

```bash
streamlit run app/main.py
```

Se abre automáticamente en `http://localhost:8501`. Desde ahí puedes escribir cualquier pregunta sobre deserción estudiantil, ejecutar el sistema y ver el reporte ejecutivo generado por los agentes, todo con una interfaz visual.

---

## Qué hace cada parte de la app

La app tiene cuatro pestañas:

**Sistema Multi-Agente** — La pestaña principal. Escribe tu consulta (o deja la que viene por defecto), haz clic en "Ejecutar" y en unos segundos aparece el reporte ejecutivo. Debajo del reporte puedes desplegar los resultados individuales de cada agente para ver qué analizó cada uno.

**Analisis Exploratorio** — Muestra las 5 gráficas del EDA a tamaño completo. Puedes navegar entre ellas con los botones de arriba. Son las mismas figuras que generó el notebook 01.

**Evaluacion del Modelo** — Muestra las métricas del Random Forest en el test set y las gráficas de evaluación: matriz de confusión, curva ROC, curva Precision-Recall e importancia de variables.

**RAG — Literatura** — Muestra los cinco documentos científicos que están indexados en ChromaDB. Puedes expandir cada uno para ver su contenido. También muestra la evaluación cuantitativa del sistema RAG.

---

## Problemas frecuentes

**El notebook 04 tarda mucho en la celda de embeddings**
Es normal la primera vez, está descargando el modelo `all-MiniLM-L6-v2`. Las siguientes veces es instantáneo porque lo guarda en caché.

**Error al correr el notebook 03 antes del 02**
Los notebooks dependen entre sí. Si el 02 no se corrió, no existen los archivos `train.csv`, `val.csv` y `test.csv` que necesita el 03. Corre siempre en orden.

**La app de Streamlit no encuentra el modelo**
Necesitas haber corrido el notebook 03 al menos una vez para que se genere el archivo `models/checkpoints/mejor_modelo.joblib`. Si no existe, la app no puede cargar las métricas ni los resultados del modelo.

---

## Estructura rápida del proyecto

```
notebooks/          → análisis paso a paso (correr en orden: 01 → 02 → 03 → 04)
app/main.py         → interfaz Streamlit
corpus/             → papers científicos usados por el agente RAG
data/               → dataset original y splits procesados
models/checkpoints/ → modelo entrenado guardado
docs/figures/       → gráficas generadas por los notebooks
```

---

Cualquier duda adicional, el README del proyecto tiene un resumen de todos los comandos principales.
