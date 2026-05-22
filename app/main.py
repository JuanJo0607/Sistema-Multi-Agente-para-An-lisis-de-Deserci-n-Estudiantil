import streamlit as st
import os
import sys
import time
import pandas as pd
import numpy as np
import joblib
from pathlib import Path
from dotenv import load_dotenv
from typing import TypedDict

# ── Rutas ─────────────────────────────────────────────────────────────────────
ROOT = Path(__file__).parent.parent
load_dotenv(ROOT / ".env")
sys.path.insert(0, str(ROOT))

from langchain_groq import ChatGroq
from langchain_core.messages import HumanMessage
from langchain_community.vectorstores import Chroma
from langchain_huggingface import HuggingFaceEmbeddings
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.document_loaders import TextLoader
from langgraph.graph import StateGraph, START, END
from sklearn.metrics import f1_score, roc_auc_score, average_precision_score

PROCESSED_PATH = ROOT / "data" / "processed"
MODELS_PATH    = ROOT / "models" / "checkpoints"
CORPUS_PATH    = ROOT / "corpus"
CHROMA_PATH    = ROOT / "chroma_db"
FIGURES_PATH   = ROOT / "docs" / "figures"

# ── Configuración de página ───────────────────────────────────────────────────
st.set_page_config(
    page_title="Sistema Multi-Agente · Deserción Estudiantil",
    page_icon=None,
    layout="wide",
    initial_sidebar_state="expanded",
)

# ── Paleta de colores ─────────────────────────────────────────────────────────
# Fondo oscuro con acento violeta (#7c3aed) y detalles en ámbar (#f59e0b)
st.markdown("""
<style>
/* Cajas de salida por agente — borde violeta */
.agent-box {
    background: #13131f;
    border-left: 4px solid #7c3aed;
    border-radius: 6px;
    padding: 1rem 1.2rem;
    margin-bottom: 0.6rem;
    font-size: 0.91rem;
    line-height: 1.6;
    white-space: pre-wrap;
    color: #e2e2f0;
}
/* Reporte ejecutivo final — borde ámbar */
.report-box {
    background: #0f0f1c;
    border: 1px solid #f59e0b;
    border-radius: 8px;
    padding: 1.4rem 1.6rem;
    font-size: 0.94rem;
    line-height: 1.7;
    white-space: pre-wrap;
    color: #f0ead6;
}
/* Etiqueta de sección pequeña */
.section-label {
    font-size: 0.72rem;
    font-weight: 700;
    letter-spacing: 0.12em;
    text-transform: uppercase;
    color: #7c3aed;
    margin-bottom: 0.3rem;
}
/* Línea separadora violeta */
hr.violet { border-color: #7c3aed33; }
</style>
""", unsafe_allow_html=True)


# ── Inicialización con caché ──────────────────────────────────────────────────
@st.cache_resource(show_spinner="Cargando modelo de embeddings...")
def init_embeddings():
    return HuggingFaceEmbeddings(
        model_name="sentence-transformers/all-MiniLM-L6-v2",
        model_kwargs={"device": "cpu"},
    )


@st.cache_resource(show_spinner="Conectando con ChromaDB...")
def init_retriever(_embeddings):
    txt_files  = list(CORPUS_PATH.glob("*.txt"))
    documentos = []
    for path in txt_files:
        loader = TextLoader(str(path), encoding="utf-8")
        docs   = loader.load()
        for d in docs:
            d.metadata["source"] = path.name
        documentos.extend(docs)

    chunks = RecursiveCharacterTextSplitter(
        chunk_size=512, chunk_overlap=64
    ).split_documents(documentos)

    if (CHROMA_PATH / "chroma.sqlite3").exists():
        vs = Chroma(persist_directory=str(CHROMA_PATH), embedding_function=_embeddings)
    else:
        vs = Chroma.from_documents(
            documents=chunks, embedding=_embeddings, persist_directory=str(CHROMA_PATH)
        )
    return vs.as_retriever(search_kwargs={"k": 4})


@st.cache_resource(show_spinner="Conectando con el LLM...")
def init_llm():
    api_key = os.getenv("GROQ_API_KEY")
    if not api_key:
        return None
    return ChatGroq(
        model="llama-3.1-8b-instant",
        groq_api_key=api_key,
        temperature=0.2,
        max_tokens=512,
    )


@st.cache_resource(show_spinner="Cargando modelo ML...")
def init_ml_resources():
    pipe    = joblib.load(MODELS_PATH / "mejor_modelo.joblib")
    test_df = pd.read_csv(PROCESSED_PATH / "test.csv")
    X_test  = test_df.drop(columns=["Dropout"])
    y_test  = test_df["Dropout"]
    y_pred  = pipe.predict(X_test)
    y_proba = pipe.predict_proba(X_test)[:, 1]

    metrics = {
        "modelo":  type(pipe.named_steps["model"]).__name__,
        "f1":      round(f1_score(y_test, y_pred), 4),
        "auc_roc": round(roc_auc_score(y_test, y_proba), 4),
        "avg_pre": round(average_precision_score(y_test, y_proba), 4),
    }

    train_df = pd.read_csv(PROCESSED_PATH / "train.csv")
    val_df   = pd.read_csv(PROCESSED_PATH / "val.csv")
    df_full  = pd.concat([train_df, val_df, test_df])
    cols_num = df_full.select_dtypes(include=[np.number]).columns.drop("Dropout").tolist()
    corr     = df_full[cols_num + ["Dropout"]].corr()["Dropout"].abs()
    top5     = corr.drop("Dropout").nlargest(5)

    eda_stats = (
        f"Dataset: {len(df_full)} estudiantes, {len(cols_num)} variables predictoras.\n"
        f"Tasa de desercion: {df_full['Dropout'].mean()*100:.1f}% (dataset desbalanceado).\n"
        f"Sin valores nulos.\n"
        f"Top 5 variables mas correlacionadas con Dropout:\n"
        + "\n".join([f"  {col} (r={val:.3f})" for col, val in top5.items()])
    )
    ml_metrics_str = (
        f"Modelo seleccionado: {metrics['modelo']}\n"
        f"F1-score (Dropout):  {metrics['f1']}\n"
        f"AUC-ROC:             {metrics['auc_roc']}\n"
        f"Average Precision:   {metrics['avg_pre']}\n"
        f"Comparado con baseline (DummyClassifier): F1=0.0, AUC=0.5"
    )
    return metrics, eda_stats, ml_metrics_str


# ── Estado del sistema ────────────────────────────────────────────────────────
class SystemState(TypedDict):
    query:        str
    eda_result:   str
    rag_result:   str
    ml_result:    str
    final_report: str


def build_graph(llm, retriever, eda_stats, ml_metrics_str):
    def agent_eda(state: SystemState) -> dict:
        prompt = (
            f"Eres un analista de datos educativo experto.\n"
            f"Basandote en los siguientes estadisticos REALES del dataset:\n\n"
            f"{eda_stats}\n\n"
            f"Responde de forma concisa (maximo 120 palabras) a:\n"
            f"{state['query']}\n\n"
            f"Estructura en 3 puntos:\n"
            f"1. Perfil del dataset\n"
            f"2. Variables mas predictivas\n"
            f"3. Implicacion principal para el modelo"
        )
        return {"eda_result": llm.invoke([HumanMessage(content=prompt)]).content}

    def agent_rag(state: SystemState) -> dict:
        docs     = retriever.invoke(state["query"])
        contexto = "\n\n---\n\n".join([d.page_content for d in docs])
        fuentes  = list({d.metadata["source"] for d in docs})
        prompt = (
            f"Eres un experto en educacion universitaria.\n"
            f"Usa UNICAMENTE el siguiente contexto de papers para responder:\n\n"
            f"{contexto}\n\n"
            f"Pregunta: {state['query']}\n\n"
            f"Responde de forma concisa (maximo 120 palabras).\n"
            f"Al final menciona: \"Fuentes: {', '.join(fuentes)}\""
        )
        return {"rag_result": llm.invoke([HumanMessage(content=prompt)]).content}

    def agent_ml(state: SystemState) -> dict:
        prompt = (
            f"Eres un cientifico de datos especializado en modelos educativos.\n"
            f"Resultados del modelo:\n\n{ml_metrics_str}\n\n"
            f"Contexto EDA:\n{state['eda_result']}\n\n"
            f"Interpreta (maximo 120 palabras):\n"
            f"1. El rendimiento es satisfactorio?\n"
            f"2. Que implica el AUC-ROC?\n"
            f"3. Que mejoras concretas recomendarias?"
        )
        return {"ml_result": llm.invoke([HumanMessage(content=prompt)]).content}

    def agent_sintetizador(state: SystemState) -> dict:
        prompt = (
            f"Eres un consultor ejecutivo de educacion universitaria.\n"
            f"Genera un REPORTE EJECUTIVO (maximo 250 palabras):\n\n"
            f"--- ANALISIS EDA ---\n{state['eda_result']}\n\n"
            f"--- LITERATURA CIENTIFICA ---\n{state['rag_result']}\n\n"
            f"--- RESULTADOS ML ---\n{state['ml_result']}\n\n"
            f"Secciones requeridas:\n"
            f"1. RESUMEN EJECUTIVO\n"
            f"2. HALLAZGOS CLAVE (3 bullets)\n"
            f"3. RENDIMIENTO DEL MODELO\n"
            f"4. RECOMENDACIONES (3 acciones)"
        )
        return {"final_report": llm.invoke([HumanMessage(content=prompt)]).content}

    builder = StateGraph(SystemState)
    builder.add_node("agent_eda",          agent_eda)
    builder.add_node("agent_rag",          agent_rag)
    builder.add_node("agent_ml",           agent_ml)
    builder.add_node("agent_sintetizador", agent_sintetizador)
    builder.add_edge(START, "agent_eda")
    builder.add_edge(START, "agent_rag")
    builder.add_edge("agent_eda", "agent_ml")
    builder.add_edge("agent_rag", "agent_ml")
    builder.add_edge("agent_ml",           "agent_sintetizador")
    builder.add_edge("agent_sintetizador", END)
    return builder.compile()


# ══════════════════════════════════════════════════════════════════════════════
#  UI
# ══════════════════════════════════════════════════════════════════════════════

st.title("Sistema Multi-Agente — Desercion Estudiantil")
st.caption("Universidad EAFIT · Proyecto Final Inteligencia Artificial 2026-1")

# ── Verificar API key ─────────────────────────────────────────────────────────
if not os.getenv("GROQ_API_KEY"):
    st.error("GROQ_API_KEY no configurada. Crea el archivo .env en la raiz del proyecto.")
    st.stop()

# ── Cargar recursos ───────────────────────────────────────────────────────────
embeddings                         = init_embeddings()
retriever                          = init_retriever(embeddings)
llm                                = init_llm()
metrics, eda_stats, ml_metrics_str = init_ml_resources()

# ── Sidebar ───────────────────────────────────────────────────────────────────
with st.sidebar:
    st.header("Modelo ML")
    st.markdown(f"**Algoritmo:** {metrics['modelo']}")
    c1, c2 = st.columns(2)
    c1.metric("F1 (Dropout)", metrics["f1"])
    c2.metric("AUC-ROC",      metrics["auc_roc"])
    st.metric("Avg Precision", metrics["avg_pre"])

    st.divider()
    st.header("Stack tecnico")
    st.markdown("""
- **LLM:** `llama-3.1-8b-instant`
- **Embeddings:** `all-MiniLM-L6-v2`
- **Vector store:** ChromaDB local
- **Agentes:** 4 (EDA · RAG · ML · Sintesis)
- **Patron:** Fan-out paralelo
    """)


# ── Pestanas principales ──────────────────────────────────────────────────────
tab_sistema, tab_eda, tab_modelo, tab_rag = st.tabs(
    ["Sistema Multi-Agente", "Analisis Exploratorio", "Evaluacion del Modelo", "RAG — Literatura"]
)

# ─────────────────────────────────────────────────────────────────────────────
with tab_sistema:
    st.subheader("Consulta al sistema")
    query = st.text_area(
        "Pregunta de analisis:",
        value=(
            "Analiza la desercion estudiantil: cuales son los principales factores "
            "de riesgo, que variables predicen mejor la desercion y que acciones puede "
            "tomar la institucion para reducirla basandose en evidencia academica."
        ),
        height=100,
    )

    run_btn = st.button("Ejecutar sistema multi-agente", type="primary", use_container_width=True)

    if run_btn:
        if not query.strip():
            st.warning("Escribe una pregunta antes de ejecutar.")
        else:
            graph  = build_graph(llm, retriever, eda_stats, ml_metrics_str)
            estado = {
                "query": query, "eda_result": "",
                "rag_result": "", "ml_result": "", "final_report": "",
            }

            progress = st.progress(0, text="Iniciando agentes...")
            status   = st.empty()

            with st.spinner(""):
                status.markdown("**Agente EDA** — analizando estadisticas del dataset...")
                progress.progress(15)
                status.markdown("**Agente RAG** — recuperando papers cientificos...")
                progress.progress(30)
                resultado = graph.invoke(estado)
                progress.progress(75)
                status.markdown("**Agente ML** — interpretando metricas del modelo...")
                time.sleep(0.3)
                progress.progress(90)
                status.markdown("**Agente Sintetizador** — generando reporte ejecutivo...")
                time.sleep(0.3)
                progress.progress(100)

            status.empty()
            progress.empty()
            st.success("Sistema ejecutado correctamente.")
            st.session_state["resultado"] = resultado

    if "resultado" in st.session_state:
        res = st.session_state["resultado"]

        st.divider()
        st.markdown('<p class="section-label">Reporte Ejecutivo Final</p>', unsafe_allow_html=True)
        st.markdown(
            f'<div class="report-box">{res["final_report"]}</div>',
            unsafe_allow_html=True,
        )

        st.divider()
        st.markdown('<p class="section-label">Detalle por agente</p>', unsafe_allow_html=True)

        with st.expander("Agente EDA — Analisis del dataset"):
            st.markdown(f'<div class="agent-box">{res["eda_result"]}</div>', unsafe_allow_html=True)

        with st.expander("Agente RAG — Literatura cientifica"):
            st.markdown(f'<div class="agent-box">{res["rag_result"]}</div>', unsafe_allow_html=True)

        with st.expander("Agente ML — Interpretacion del modelo"):
            st.markdown(f'<div class="agent-box">{res["ml_result"]}</div>', unsafe_allow_html=True)

# ─────────────────────────────────────────────────────────────────────────────
with tab_eda:
    st.subheader("Analisis Exploratorio de Datos")

    FIGURAS_EDA = [
        ("Distribucion de la variable objetivo",  "01_distribucion_objetivo.png",
         "Distribucion de clases Dropout vs No Dropout. Tasa de desercion: 32.1% — dataset desbalanceado."),
        ("Boxplots de variables con outliers",     "02_boxplots_outliers.png",
         "Comparacion de las 6 variables con mas outliers entre estudiantes que desertaron y los que no."),
        ("Variables academicas del 1er semestre",  "03_variables_academicas.png",
         "Histogramas superpuestos: los desertores (rojo) acumulan muchas menos unidades aprobadas."),
        ("Matriz de correlacion",                  "04_correlacion.png",
         "Top 14 variables con mayor correlacion absoluta con Dropout. Las academicas dominan."),
        ("Variables socioeconomicas",              "05_socioeconomicas.png",
         "Tasa de desercion segun beca, deuda, pago de matricula y genero."),
    ]

    nombres = [f[0] for f in FIGURAS_EDA]
    seleccion = st.radio("Seleccionar figura:", nombres, horizontal=True)

    for nombre, archivo, descripcion in FIGURAS_EDA:
        if seleccion == nombre:
            ruta = FIGURES_PATH / archivo
            if ruta.exists():
                st.image(str(ruta), use_column_width=True)
                st.caption(descripcion)
            break

# ─────────────────────────────────────────────────────────────────────────────
with tab_modelo:
    st.subheader("Evaluacion final en Test Set")

    c1, c2, c3 = st.columns(3)
    c1.metric("F1-score (Dropout)", metrics["f1"],
              delta=f"+{metrics['f1']:.2f} vs baseline 0.00")
    c2.metric("AUC-ROC",            metrics["auc_roc"],
              delta=f"+{metrics['auc_roc'] - 0.5:.2f} vs baseline 0.50")
    c3.metric("Avg Precision",      metrics["avg_pre"])

    st.divider()
    ca, cb = st.columns(2)
    for path, caption, col in [
        (FIGURES_PATH / "08_evaluacion_final.png",  "Matriz de confusion · Curva ROC · Curva PR", ca),
        (FIGURES_PATH / "09_feature_importance.png", "Top 15 variables mas predictivas",           cb),
        (FIGURES_PATH / "07_curvas_aprendizaje.png", "Curvas de aprendizaje",                      ca),
        (FIGURES_PATH / "06_tabla_modelos.png",      "Comparacion de modelos",                     cb),
    ]:
        if path.exists():
            col.image(str(path), caption=caption, use_column_width=True)

# ─────────────────────────────────────────────────────────────────────────────
with tab_rag:
    st.subheader("Corpus RAG — Papers indexados")
    corpus_files = sorted(CORPUS_PATH.glob("*.txt"))

    st.info(f"**{len(corpus_files)} documentos** indexados en ChromaDB con embeddings `all-MiniLM-L6-v2`")

    for path in corpus_files:
        with st.expander(path.stem.replace("_", " ").title()):
            st.text(path.read_text(encoding="utf-8")[:800] + "\n...")

    st.divider()
    fig_rag = FIGURES_PATH / "10_evaluacion_rag.png"
    if fig_rag.exists():
        st.subheader("Evaluacion cuantitativa del RAG")
        st.image(str(fig_rag), caption="Accuracy del RAG: 10/10 (100%)",
                 use_column_width=False, width=600)
