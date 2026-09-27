"""Interfaz Streamlit de entrenamiento/calibración del Motor IA.

Independiente de la API de producción (FastAPI): se ejecuta solo cuando el
equipo necesita disparar EDA, entrenamiento, tuning o revisar comparaciones
estadísticas. Nunca se expone en producción.

Uso:
    streamlit run backend/training/streamlit_app.py

Esta fase (5) implementa la pestaña de EDA. Las pestañas de Entrenamiento,
Comparación de arquitecturas y Estadística se añaden en las Fases 6-8 sin
modificar esta estructura base.
"""

from __future__ import annotations

import json

import streamlit as st

from backend.training import synthetic_wesad, wesad_loader  # noqa: F401 - registran los datasets
from backend.training.dataset_loader import DATASET_REGISTRY, get_loader
from backend.training.eda import run_eda
from backend.utils.config import get_settings

st.set_page_config(page_title="Motor IA — Gemelo Digital de Evacuación Minera", layout="wide")

settings = get_settings()

st.title("Motor IA — Entrenamiento y Calibración")
st.caption(
    "Interfaz independiente de la API de producción. Aquí se dispara la descarga, "
    "el EDA, el entrenamiento, la comparación de arquitecturas y las pruebas estadísticas."
)

tab_eda, tab_training, tab_comparison, tab_stats = st.tabs(
    ["📊 EDA", "🧠 Entrenamiento (Fase 6)", "⚖️ Comparación (Fase 7)", "📈 Estadística (Fase 7)"]
)

with tab_eda:
    st.header("Análisis Exploratorio de Datos (EDA)")

    dataset_names = list(DATASET_REGISTRY.keys())
    selected_dataset = st.selectbox("Dataset", dataset_names, index=0)
    loader = get_loader(selected_dataset)

    is_synthetic = selected_dataset != "wesad"
    if is_synthetic:
        st.warning(
            "⚠️ Este dataset es SINTÉTICO. Sirve solo para validar el pipeline; "
            "ninguna cifra generada con él debe usarse en el artículo."
        )

    available = loader.is_available_locally()
    st.write(f"**Disponible localmente:** {'✅ Sí' if available else '❌ No'}")

    if not available:
        st.info(
            "Ejecute `python -m backend.training.download_wesad` en una máquina con "
            "acceso a internet, y coloque los datos en `backend/data/raw/WESAD/`."
        )
    else:
        subjects = loader.list_subjects()
        st.write(f"**Sujetos encontrados:** {len(subjects)} — {subjects}")

        subject_limit = st.slider(
            "Límite de sujetos a analizar (para EDA rápido)",
            min_value=1,
            max_value=len(subjects),
            value=min(5, len(subjects)),
        )

        if st.button("▶️ Ejecutar EDA", type="primary"):
            with st.spinner("Generando EDA…"):
                summary = run_eda(loader, subject_limit=subject_limit)
            st.success(f"EDA completado. Artefactos en: {summary['artifacts_dir']}")
            st.session_state["last_eda_summary"] = summary

        summary = st.session_state.get("last_eda_summary")
        if summary and summary["dataset_name"] == selected_dataset:
            from pathlib import Path

            artifacts_dir = Path(summary["artifacts_dir"])

            st.subheader("Distribución de clases")
            st.image(str(artifacts_dir / "class_distribution.png"))

            st.subheader("Señales de ejemplo")
            st.image(str(artifacts_dir / "example_signals.png"))

            st.subheader("Correlación entre features")
            st.image(str(artifacts_dir / "feature_correlation_heatmap.png"))

            st.subheader("Reporte de calidad")
            st.json(summary["quality_report"])

            with open(artifacts_dir / "artifact_detection_report.json", encoding="utf-8") as f:
                st.subheader("Detección de artefactos por sujeto/canal")
                st.json(json.load(f))

with tab_training:
    st.info("La interfaz de entrenamiento, CV y tuning de hiperparámetros se implementa en la Fase 6.")

with tab_comparison:
    st.info("La comparación cuantitativa de arquitecturas y estrategias se implementa en la Fase 7.")

with tab_stats:
    st.info("Las pruebas estadísticas (Wilcoxon, Friedman, Nemenyi) se implementan en la Fase 7.")
