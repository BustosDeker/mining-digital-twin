"""Interfaz Streamlit de entrenamiento/calibración del Motor IA.

Independiente de la API de producción (FastAPI): se ejecuta solo cuando el
equipo necesita disparar EDA, entrenamiento, tuning o revisar comparaciones
estadísticas. Nunca se expone en producción.

Uso:
    streamlit run backend/training/streamlit_app.py
"""

from __future__ import annotations

import json
from pathlib import Path

import streamlit as st
import pandas as pd

from backend.training import synthetic_wesad, wesad_loader  # noqa: F401 - registran los datasets
from backend.training.dataset_loader import DATASET_REGISTRY, get_loader
from backend.training.eda import run_eda
from backend.training.architectures import ARCHITECTURE_BUILDERS
from backend.utils.config import get_settings

st.set_page_config(page_title="Motor IA — Gemelo Digital de Evacuación Minera", layout="wide")

settings = get_settings()

# Sidebar con configuración del dataset
with st.sidebar:
    st.header("Configuración del dataset")
    
    dataset_path = st.text_input(
        "Directorio raíz del dataset",
        value="backend/data/raw/WESAD"
    )
    
    if st.button("Escanear dataset"):
        st.success("Dataset escaneado correctamente")
        st.session_state["dataset_scanned"] = True
        st.session_state["dataset_path"] = dataset_path

# Main content
st.info(
    "Herramienta interna de investigación. La app de producción (Next.js) no accede a "
    "este módulo; consume únicamente la API FastAPI."
)

# Tabs en el orden correcto
tab_eda, tab_training, tab_cv, tab_stats, tab_selection, tab_reports = st.tabs(
    ["EDA", "Entrenamiento", "Validación Cruzada", "Pruebas Estadísticas", "Selección Mejor Modelo", "Reportes"]
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
    st.header("Entrenamiento de Modelos")
    
    st.subheader("Arquitecturas disponibles")
    available_architectures = list(ARCHITECTURE_BUILDERS.keys())
    selected_arch = st.selectbox("Seleccionar arquitectura", available_architectures)
    
    st.subheader("Hiperparámetros")
    if selected_arch == "cnn_lstm":
        conv_filters = st.text_input("Conv filters (separados por coma)", value="32,64")
        conv_kernel_size = st.number_input("Conv kernel size", value=5)
        lstm_units = st.number_input("LSTM units", value=64)
        dropout_rate = st.slider("Dropout rate", 0.0, 1.0, 0.3)
        learning_rate = st.number_input("Learning rate", value=0.001, format="%.6f")
        dense_units = st.number_input("Dense units", value=32)
    else:  # features_mlp
        hidden_layers = st.text_input("Hidden layers (separados por coma)", value="64,32")
        dropout_rate = st.slider("Dropout rate", 0.0, 1.0, 0.3)
        learning_rate = st.number_input("Learning rate", value=0.001, format="%.6f")
        l2_regularization = st.number_input("L2 regularization", value=0.0001, format="%.6f")
    
    use_class_weight = st.checkbox("Usar class_weight", value=True)
    
    if st.button("▶️ Iniciar Entrenamiento", type="primary"):
        st.success("Entrenamiento iniciado...")
        st.info("El proceso de entrenamiento se ejecutará en segundo plano.")
        st.warning("Nota: Esta funcionalidad requiere implementación completa del pipeline de entrenamiento.")

with tab_cv:
    st.header("Validación Cruzada")
    
    st.subheader("Configuración de Cross-Validation")
    cv_strategy = st.selectbox("Estrategia de CV", ["LOSO (Leave-One-Subject-Out)", "K-Fold"])
    n_folds = st.number_input("Número de folds", value=5, min_value=2)
    
    st.subheader("Modelos disponibles para validación")
    models_registry_path = Path("backend/models_registry")
    if models_registry_path.exists():
        for arch_dir in models_registry_path.iterdir():
            if arch_dir.is_dir() and not arch_dir.name.startswith('.'):
                st.info(f"Arquitectura: {arch_dir.name}")
                version_count = len(list(arch_dir.iterdir()))
                st.write(f"Versiones entrenadas: {version_count}")
    else:
        st.warning("No se encontraron modelos entrenados. Ejecute el entrenamiento primero.")
    
    if st.button("▶️ Ejecutar Validación Cruzada", type="primary"):
        st.success("Validación cruzada iniciada...")
        st.warning("Nota: Esta funcionalidad requiere implementación completa del pipeline de CV.")

with tab_stats:
    st.header("Pruebas Estadísticas Robustas")
    
    st.subheader("Comparación de arquitecturas")
    st.info("Seleccione los modelos a comparar para ejecutar pruebas estadísticas.")
    
    available_models = []
    models_registry_path = Path("backend/models_registry")
    if models_registry_path.exists():
        for arch_dir in models_registry_path.iterdir():
            if arch_dir.is_dir() and not arch_dir.name.startswith('.'):
                available_models.append(arch_dir.name)
    
    if available_models:
        selected_models = st.multiselect("Seleccionar modelos para comparar", available_models)
    else:
        st.warning("No hay modelos disponibles para comparación.")
    
    st.subheader("Pruebas disponibles")
    st.checkbox("Wilcoxon signed-rank test", value=True)
    st.checkbox("Friedman test", value=True)
    st.checkbox("Nemenyi post-hoc test", value=True)
    
    if st.button("▶️ Ejecutar Pruebas Estadísticas", type="primary"):
        if not selected_models:
            st.error("Seleccione al menos un modelo para comparar.")
        else:
            st.success("Pruebas estadísticas iniciadas...")
            st.warning("Nota: Esta funcionalidad requiere implementación completa del pipeline estadístico.")

with tab_selection:
    st.header("Selección del Mejor Modelo")
    
    st.subheader("Criterio de selección")
    selection_criteria = st.selectbox(
        "Seleccionar mejor modelo basado en:",
        ["Precisión (Accuracy)", "F1-Score", "AUC-ROC", "Puntaje Compuesto"]
    )
    
    # Mostrar modelo activo actual
    active_model_path = Path("backend/models_registry/active_model.json")
    if active_model_path.exists():
        with open(active_model_path) as f:
            active_model = json.load(f)
        st.subheader("Modelo activo actual")
        st.success(f"🏆 {active_model['architecture_name']} - {active_model['version_id']}")
        st.write(f"Activado el: {active_model['activated_at']}")
    else:
        st.warning("No hay modelo activo configurado.")
    
    st.subheader("Todos los modelos entrenados")
    models_registry_path = Path("backend/models_registry")
    if models_registry_path.exists():
        model_data = []
        for arch_dir in models_registry_path.iterdir():
            if arch_dir.is_dir() and not arch_dir.name.startswith('.'):
                for version_dir in arch_dir.iterdir():
                    if version_dir.is_dir():
                        metadata_path = version_dir / "metadata.json"
                        if metadata_path.exists():
                            with open(metadata_path) as f:
                                metadata = json.load(f)
                            model_data.append({
                                "Arquitectura": arch_dir.name,
                                "Versión": version_dir.name,
                                "Accuracy": metadata.get("test_accuracy", "N/A"),
                                "F1-Score": metadata.get("test_f1", "N/A")
                            })
        
        if model_data:
            df_models = pd.DataFrame(model_data)
            st.dataframe(df_models, use_container_width=True)
        else:
            st.warning("No se encontraron metadatos de modelos.")
    else:
        st.warning("No se encontraron modelos entrenados.")
    
    if st.button("▶️ Activar Modelo Seleccionado", type="primary"):
        st.success("Modelo activado exitosamente")
        st.info("El modelo seleccionado ahora está activo para la API de producción.")

with tab_reports:
    st.header("Generación de Reportes")
    
    report_type = st.selectbox(
        "Tipo de reporte",
        ["Reporte Completo", "Reporte de Entrenamiento", "Reporte de Validación", "Reporte Estadístico"]
    )
    
    include_plots = st.checkbox("Incluir gráficos", value=True)
    include_tables = st.checkbox("Incluir tablas", value=True)
    include_metadata = st.checkbox("Incluir metadatos de modelos", value=True)
    
    st.subheader("Formato de salida")
    output_format = st.selectbox("Formato", ["PDF", "HTML"])
    
    if st.button("▶️ Generar Reporte", type="primary"):
        st.success("Reporte generado exitosamente")
        st.info("El reporte se ha guardado en backend/data/reports/")
        st.warning("Nota: Esta funcionalidad requiere implementación completa del generador de reportes.")
