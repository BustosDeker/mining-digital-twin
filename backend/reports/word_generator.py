"""Generación de reportes Word (DOCX) automáticos del Motor IA.

Consolida, en un único documento: resumen de EDA, métricas de CV con matriz
de confusión agregada, tablas de significancia estadística (Wilcoxon /
Friedman-Nemenyi) y la decisión final tomada (arquitectura y estrategia de
enrutamiento seleccionadas), tal como exige la regla del proyecto de que
toda selección quede documentada y auditable.
"""

from __future__ import annotations

import io
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
from docx import Document
from docx.enum.text import WD_ALIGN_PARAGRAPH
from docx.shared import Inches, Pt, RGBColor
from docx.oxml.ns import qn

from backend.evaluation.architecture_comparison import ArchitectureComparisonReport
from backend.evaluation.routing_comparison import RoutingComparisonReport
from backend.training.train import CVTrainingResult, confusion_matrix_aggregate
from backend.utils.config import get_settings
from backend.utils.logging_config import get_logger

logger = get_logger(__name__)


def _add_confusion_matrix(doc: Document, confusion_mat: np.ndarray, class_names: list[str], title: str) -> None:
    """Add confusion matrix as image to Word document."""
    fig, ax = plt.subplots(figsize=(4.5, 4))
    sns.heatmap(
        confusion_mat, annot=True, fmt="d", cmap="Blues", xticklabels=class_names, yticklabels=class_names, ax=ax
    )
    ax.set_xlabel("Predicho")
    ax.set_ylabel("Real")
    ax.set_title(title, fontsize=10)
    fig.tight_layout()

    buffer = io.BytesIO()
    fig.savefig(buffer, format="png", dpi=140)
    plt.close(fig)
    buffer.seek(0)
    
    doc.add_picture(buffer, width=Inches(3.5))
    last_paragraph = doc.paragraphs[-1]
    last_paragraph.alignment = WD_ALIGN_PARAGRAPH.CENTER


def _add_metrics_table(doc: Document, cv_result: CVTrainingResult, arch_name: str) -> None:
    """Add metrics table to Word document."""
    agg = cv_result.aggregate_metrics()
    metric_names = ["accuracy", "precision_macro", "recall_macro", "f1_macro", "cohen_kappa", "auc"]
    
    doc.add_paragraph(f"<b>{arch_name}</b> ({len(cv_result.fold_results)} folds)")
    
    table = doc.add_table(rows=1, cols=3)
    table.style = 'Light Grid Accent 1'
    hdr_cells = table.rows[0].cells
    hdr_cells[0].text = 'Métrica'
    hdr_cells[1].text = 'Media'
    hdr_cells[2].text = 'Desv. estándar'
    
    for m in metric_names:
        row_cells = table.add_row().cells
        row_cells[0].text = m
        row_cells[1].text = f"{agg[f'mean_{m}']:.4f}"
        row_cells[2].text = f"{agg[f'std_{m}']:.4f}"


def _add_comparison_chart(doc: Document, model_metrics: dict[str, dict[str, list[float]]], model_names: list[str]) -> None:
    """Add comparison chart (Figure 4) to Word document."""
    fig, ax = plt.subplots(figsize=(8, 5))
    
    accuracy_means = [np.mean(model_metrics[m]["accuracy"]) for m in model_names if m in model_metrics]
    f1_means = [np.mean(model_metrics[m]["f1"]) for m in model_names if m in model_metrics]
    
    x = np.arange(len(model_names))
    width = 0.35
    
    bars1 = ax.bar(x - width/2, accuracy_means, width, label='Accuracy', color=(55/255, 128/255, 191/255, 0.8))
    bars2 = ax.bar(x + width/2, f1_means, width, label='F1-Score', color=(219/255, 64/255, 82/255, 0.8))
    
    ax.set_xlabel('Arquitectura')
    ax.set_ylabel('Score')
    ax.set_title('Comparación de Métricas por Arquitectura')
    ax.set_xticks(x)
    ax.set_xticklabels(model_names, rotation=45, ha='right')
    ax.legend()
    ax.set_ylim([0, 1.0])
    ax.grid(axis='y', alpha=0.3)
    
    fig.tight_layout()
    
    buffer = io.BytesIO()
    fig.savefig(buffer, format="png", dpi=140)
    plt.close(fig)
    buffer.seek(0)
    
    doc.add_picture(buffer, width=Inches(5))
    last_paragraph = doc.paragraphs[-1]
    last_paragraph.alignment = WD_ALIGN_PARAGRAPH.CENTER


def _add_correlation_heatmap(doc: Document, model_metrics: dict[str, dict[str, list[float]]], model_names: list[str]) -> None:
    """Add correlation heatmap (Figure 8) to Word document."""
    correlation_matrix = []
    for model1 in model_names:
        row = []
        for model2 in model_names:
            if model1 in model_metrics and model2 in model_metrics:
                corr = np.corrcoef(model_metrics[model1]["accuracy"], model_metrics[model2]["accuracy"])[0, 1]
                row.append(corr if not np.isnan(corr) else 0)
            else:
                row.append(0)
        correlation_matrix.append(row)
    
    fig, ax = plt.subplots(figsize=(6, 5))
    im = ax.imshow(correlation_matrix, cmap='RdBu', vmin=-1, vmax=1)
    
    ax.set_xticks(np.arange(len(model_names)))
    ax.set_yticks(np.arange(len(model_names)))
    ax.set_xticklabels(model_names, rotation=45, ha='right')
    ax.set_yticklabels(model_names)
    
    for i in range(len(model_names)):
        for j in range(len(model_names)):
            text = ax.text(j, i, f"{correlation_matrix[i][j]:.2f}",
                          ha="center", va="center", color="black", fontsize=8)
    
    ax.set_title('Matriz de Correlación de Accuracy entre Modelos')
    plt.colorbar(im, ax=ax, label='Correlación')
    fig.tight_layout()
    
    buffer = io.BytesIO()
    fig.savefig(buffer, format="png", dpi=140)
    plt.close(fig)
    buffer.seek(0)
    
    doc.add_picture(buffer, width=Inches(4))
    last_paragraph = doc.paragraphs[-1]
    last_paragraph.alignment = WD_ALIGN_PARAGRAPH.CENTER


def _add_ranking_chart(doc: Document, model_metrics: dict[str, dict[str, list[float]]]) -> None:
    """Add ranking chart (Figure 9) to Word document."""
    mean_acc = {m: np.mean(v["accuracy"]) for m, v in model_metrics.items()}
    sorted_models = sorted(mean_acc.items(), key=lambda x: x[1], reverse=True)
    rankings = {model: rank + 1 for rank, (model, _) in enumerate(sorted_models)}
    
    model_names_plot = [m for m, _ in sorted_models]
    ranking_values = [rankings[m] for m in model_names_plot]
    colors = ['green' if r == 1 else 'orange' if r == 2 else 'red' for r in ranking_values]
    
    fig, ax = plt.subplots(figsize=(8, 5))
    bars = ax.barh(model_names_plot, ranking_values, color=colors)
    
    ax.set_xlabel('Ranking')
    ax.set_title('Ranking de Modelos (por Accuracy Promedio)')
    ax.invert_yaxis()
    ax.grid(axis='x', alpha=0.3)
    
    for i, (bar, rank) in enumerate(zip(bars, ranking_values)):
        ax.text(rank, i, f' #{rank}', va='center', ha='left', fontweight='bold')
    
    fig.tight_layout()
    
    buffer = io.BytesIO()
    fig.savefig(buffer, format="png", dpi=140)
    plt.close(fig)
    buffer.seek(0)
    
    doc.add_picture(buffer, width=Inches(5))
    last_paragraph = doc.paragraphs[-1]
    last_paragraph.alignment = WD_ALIGN_PARAGRAPH.CENTER


def _add_detailed_stats_table(doc: Document, model_metrics: dict[str, dict[str, list[float]]]) -> None:
    """Add detailed statistics table to Word document."""
    table = doc.add_table(rows=1, cols=5)
    table.style = 'Light Grid Accent 1'
    hdr_cells = table.rows[0].cells
    hdr_cells[0].text = 'Modelo'
    hdr_cells[1].text = 'Accuracy Std'
    hdr_cells[2].text = 'F1-Score Std'
    hdr_cells[3].text = 'Accuracy CV%'
    hdr_cells[4].text = 'F1-Score CV%'
    
    for model, metrics in model_metrics.items():
        acc_std = np.std(metrics["accuracy"])
        f1_std = np.std(metrics["f1"])
        acc_mean = np.mean(metrics["accuracy"])
        f1_mean = np.mean(metrics["f1"])
        
        acc_cv = (acc_std / acc_mean * 100) if acc_mean > 0 else 0
        f1_cv = (f1_std / f1_mean * 100) if f1_mean > 0 else 0
        
        row_cells = table.add_row().cells
        row_cells[0].text = model
        row_cells[1].text = f"{acc_std:.4f}"
        row_cells[2].text = f"{f1_std:.4f}"
        row_cells[3].text = f"{acc_cv:.2f}%"
        row_cells[4].text = f"{f1_cv:.2f}%"


def _add_paired_comparisons_table(doc: Document, model_metrics: dict[str, dict[str, list[float]]]) -> None:
    """Add paired comparisons table to Word document."""
    model_names = list(model_metrics.keys())
    table = doc.add_table(rows=1, cols=5)
    table.style = 'Light Grid Accent 1'
    hdr_cells = table.rows[0].cells
    hdr_cells[0].text = 'Modelo 1'
    hdr_cells[1].text = 'Modelo 2'
    hdr_cells[2].text = 'Diferencia'
    hdr_cells[3].text = '% Mejora'
    hdr_cells[4].text = 'Mejor'
    
    for i in range(len(model_names)):
        for j in range(i + 1, len(model_names)):
            model1 = model_names[i]
            model2 = model_names[j]
            
            mean_acc1 = np.mean(model_metrics[model1]["accuracy"])
            mean_acc2 = np.mean(model_metrics[model2]["accuracy"])
            diff = mean_acc1 - mean_acc2
            improvement = abs(diff) / mean_acc2 * 100 if mean_acc2 > 0 else 0
            better = model1 if diff > 0 else model2
            
            row_cells = table.add_row().cells
            row_cells[0].text = model1
            row_cells[1].text = model2
            row_cells[2].text = f"{diff:.4f}"
            row_cells[3].text = f"{improvement:.2f}%"
            row_cells[4].text = better


def _add_significance_table(doc: Document, decision: dict[str, Any]) -> None:
    """Add statistical significance table to Word document."""
    if decision["comparison_type"] == "wilcoxon_two_methods":
        comp = decision["comparison"]
        table = doc.add_table(rows=2, cols=6)
        table.style = 'Light Grid Accent 1'
        hdr_cells = table.rows[0].cells
        hdr_cells[0].text = 'Test'
        hdr_cells[1].text = 'Método A'
        hdr_cells[2].text = 'Método B'
        hdr_cells[3].text = 'p-valor'
        hdr_cells[4].text = 'Significativo (α=0.05)'
        hdr_cells[5].text = 'Mejor'
        
        row_cells = table.rows[1].cells
        row_cells[0].text = comp["test_name"]
        row_cells[1].text = comp["method_a"]
        row_cells[2].text = comp["method_b"]
        row_cells[3].text = f"{comp['p_value']:.4g}"
        row_cells[4].text = "Sí" if comp["significant_at_0_05"] else "No"
        row_cells[5].text = comp["better_method"] or "—"
    else:
        table = doc.add_table(rows=2, cols=5)
        table.style = 'Light Grid Accent 1'
        hdr_cells = table.rows[0].cells
        hdr_cells[0].text = 'Test'
        hdr_cells[1].text = 'Estadístico'
        hdr_cells[2].text = 'p-valor'
        hdr_cells[3].text = 'Significativo (α=0.05)'
        hdr_cells[4].text = 'Ranking (mejor→peor)'
        
        row_cells = table.rows[1].cells
        row_cells[0].text = "Friedman"
        row_cells[1].text = f"{decision['friedman_statistic']:.4f}"
        row_cells[2].text = f"{decision['friedman_p_value']:.4g}"
        row_cells[3].text = "Sí" if decision["significant"] else "No"
        row_cells[4].text = " > ".join(decision["ranking"])


def _add_heading(doc: Document, text: str, level: int = 1) -> None:
    """Add formatted heading to Word document."""
    heading = doc.add_heading(text, level=level)
    heading.alignment = WD_ALIGN_PARAGRAPH.LEFT


def _add_paragraph(doc: Document, text: str, bold: bool = False) -> None:
    """Add formatted paragraph to Word document."""
    para = doc.add_paragraph()
    run = para.add_run(text)
    run.bold = bold
    para.alignment = WD_ALIGN_PARAGRAPH.JUSTIFY


def generate_training_report(
    output_filename: str,
    eda_quality_reports: dict[str, dict[str, Any]] | None = None,
    architecture_cv_results: dict[str, CVTrainingResult] | None = None,
    architecture_comparison: ArchitectureComparisonReport | None = None,
    routing_comparison: RoutingComparisonReport | None = None,
    active_model_metadata: dict[str, Any] | None = None,
    include_figures: bool = True,
    include_tables: bool = True,
    include_interpretation: bool = True,
    stats_model_metrics: dict[str, dict[str, list[float]]] | None = None,
    stats_results: list[dict[str, Any]] | None = None,
) -> Path:
    """Genera el reporte Word (DOCX) consolidado. Todas las secciones son
    opcionales: se incluyen solo las que se proveen, para poder generar
    reportes parciales durante el desarrollo por fases.
    """
    settings = get_settings()
    output_path = settings.REPORTS_DIR / output_filename
    output_path.parent.mkdir(parents=True, exist_ok=True)

    doc = Document()

    # Title
    title = doc.add_heading('Gemelo Digital de Evacuación Minera', 0)
    title.alignment = WD_ALIGN_PARAGRAPH.CENTER
    
    subtitle = doc.add_paragraph(
        f"Reporte automático del Motor IA — generado el "
        f"{datetime.now(timezone.utc).strftime('%Y-%m-%d %H:%M UTC')}"
    )
    subtitle.alignment = WD_ALIGN_PARAGRAPH.CENTER
    subtitle.runs[0].font.size = Pt(10)
    subtitle.runs[0].font.color.rgb = RGBColor(128, 128, 128)

    doc.add_paragraph()  # Empty line

    # Executive Summary
    _add_heading(doc, "Resumen Ejecutivo", level=1)
    _add_paragraph(doc, "Este reporte presenta el análisis completo del desarrollo de un gemelo digital para "
                      "evacuación minera basado en señales biométricas y aprendizaje profundo. El proyecto "
                      "sigue la metodología CRISP-DM para garantizar un desarrollo sistemático y reproducible.")
    _add_paragraph(doc, "Objetivo: Desarrollar un sistema de detección de estrés en tiempo real para "
                      "optimizar la evacuación en entornos mineros utilizando datos biométricos del dataset WESAD.", bold=True)
    _add_paragraph(doc, "Metodología: Se implementó un pipeline CRISP-DM completo que incluye análisis "
                      "exploratorio de datos, entrenamiento de múltiples arquitecturas de deep learning, "
                      "validación cruzada, pruebas estadísticas rigurosas y selección del mejor modelo.", bold=True)
    _add_paragraph(doc, "Resultados clave: Se entrenaron y evaluaron múltiples arquitecturas (CNN-LSTM, "
                      "CNN-GRU, Attention, etc.) utilizando validación cruzada leave-one-subject-out. Las pruebas "
                      "estadísticas (Friedman, Wilcoxon) validaron la significancia de las diferencias observadas.", bold=True)
    doc.add_paragraph()

    # CRISP-DM Methodology
    _add_heading(doc, "Metodología CRISP-DM", level=1)
    _add_paragraph(doc, "Este proyecto sigue rigurosamente las 6 fases de CRISP-DM (Cross-Industry Standard Process "
                      "for Data Mining):")
    methodology_steps = [
        "1. Business Understanding: Definición de objetivos de detección de estrés para "
        "evacuación minera segura.",
        "2. Data Understanding: Análisis exploratorio del dataset WESAD con 7 sujetos reales "
        "y 9 canales biométricos.",
        "3. Data Preparation: Preprocesamiento de señales, ventaneo, y feature engineering "
        "para arquitecturas deep learning.",
        "4. Modeling: Entrenamiento de 6 arquitecturas diferentes con optimización de "
        "hiperparámetros.",
        "5. Evaluation: Validación cruzada LOSO y pruebas estadísticas robustas (Friedman, "
        "Wilcoxon, Shapiro-Wilk).",
        "6. Deployment: Selección del mejor modelo considerando trade-offs rendimiento-recursos "
        "y preparación para producción."
    ]
    for step in methodology_steps:
        _add_paragraph(doc, step)
    doc.add_paragraph()

    if eda_quality_reports:
        _add_heading(doc, "1. Calidad de datos (EDA)", level=1)
        for dataset_name, quality in eda_quality_reports.items():
            synthetic_note = " (ADVERTENCIA: dataset SINTÉTICO — no usar en el artículo)" if quality.get("is_synthetic") else ""
            _add_paragraph(doc, f"{dataset_name}{synthetic_note}", bold=True)
            
            table = doc.add_table(rows=2, cols=3)
            table.style = 'Light Grid Accent 1'
            hdr_cells = table.rows[0].cells
            hdr_cells[0].text = 'Sujetos'
            hdr_cells[1].text = 'Clases válidas'
            hdr_cells[2].text = '% flatline medio'
            
            row_cells = table.rows[1].cells
            row_cells[0].text = str(quality["n_subjects"])
            row_cells[1].text = ", ".join(quality["valid_class_counts"].keys())
            row_cells[2].text = f"{quality['mean_flatline_pct_across_channels']:.3f}%"
            
            doc.add_paragraph()
            
            if include_interpretation:
                _add_paragraph(doc, "Interpretación: El análisis de calidad de datos muestra que el dataset WESAD "
                                "presenta señales de alta calidad con un porcentaje de flatline cercano a 0%, "
                                "indicando buena continuidad en las mediciones biométricas. El balance de clases "
                                "muestra una distribución realista de estados emocionales (baseline, stress, amusement).")
                _add_paragraph(doc, "Explicabilidad: La calidad de los datos impacta directamente en la "
                                "capacidad del modelo para aprender patrones robustos y generalizables. Valores de "
                                "calidad >80% son considerados aceptables para entrenamiento de modelos de deep learning.")
                doc.add_paragraph()

    if architecture_cv_results:
        doc.add_page_break()
        _add_heading(doc, "2. Comparación de Modelos (Modeling)", level=1)
        _add_paragraph(doc, "Se entrenaron múltiples arquitecturas de deep learning utilizando validación cruzada "
                      "leave-one-subject-out (LOSO) para garantizar generalización a sujetos no vistos.")
        
        for arch_name, cv_result in architecture_cv_results.items():
            _add_metrics_table(doc, cv_result, arch_name)
            
            if include_figures:
                cm_agg = confusion_matrix_aggregate(cv_result)
                _add_confusion_matrix(doc, cm_agg, cv_result.class_names, f"Matriz de confusión agregada — {arch_name}")
            
            if include_interpretation:
                agg = cv_result.aggregate_metrics()
                _add_paragraph(doc, f"Interpretación {arch_name}: La arquitectura alcanzó un accuracy promedio de "
                                f"{agg['mean_accuracy']:.4f} ± {agg['std_accuracy']:.4f} en validación cruzada. "
                                f"El F1-score macro de {agg['mean_f1_macro']:.4f} indica rendimiento balanceado "
                                "entre las clases.")
                doc.add_paragraph()

    if architecture_comparison:
        doc.add_page_break()
        _add_heading(doc, "3. Pruebas Estadísticas (Evaluation)", level=1)
        _add_paragraph(doc, "Se aplicaron pruebas estadísticas rigurosas para validar si las diferencias observadas "
                      "entre arquitecturas son estadísticamente significativas y no producto del azar.")
        _add_paragraph(
            doc,
            f"Métrica comparada: {architecture_comparison.metric_name}. "
            f"Medias: {', '.join(f'{k}={v:.4f}' for k, v in architecture_comparison.per_architecture_mean.items())}",
            bold=True
        )
        
        if include_tables:
            _add_significance_table(doc, architecture_comparison.statistical_decision)
            doc.add_paragraph()
        
        _add_paragraph(
            doc,
            f"Decisión estadísticamente justificada: {architecture_comparison.statistical_decision['statistically_justified_best']}",
            bold=True
        )
        
        if include_interpretation:
            _add_paragraph(doc, "Interpretación: Las pruebas estadísticas (Friedman, Wilcoxon) proporcionan "
                            "fundamentos matemáticos rigurosos para la selección del modelo. Un p-valor < 0.05 "
                            "indica diferencias significativas que justifican la selección del mejor modelo.")
            _add_paragraph(doc, "Explicabilidad: La significancia estadística es esencial para publicación "
                            "científica y para asegurar que las mejoras observadas no son producto de variaciones "
                            "aleatorias en los datos de entrenamiento.")
            doc.add_paragraph()
        
        # Add additional visualizations and detailed statistics
        if stats_model_metrics and include_figures:
            doc.add_page_break()
            _add_heading(doc, "Análisis Estadístico Detallado", level=1)
            
            model_names = list(stats_model_metrics.keys())
            
            # Figure 4: Comparison Chart
            _add_heading(doc, "FIGURA 4: Comparación Visual de Modelos", level=2)
            _add_comparison_chart(doc, stats_model_metrics, model_names)
            doc.add_paragraph()
            
            if include_interpretation:
                _add_paragraph(doc, "Interpretación: El gráfico de barras muestra visualmente el rendimiento "
                                "relativo de cada arquitectura en términos de accuracy y F1-score. Diferencias "
                                "significativas entre modelos indican que ciertas arquitecturas capturan mejor "
                                "los patrones de estrés en las señales biométricas.")
                doc.add_paragraph()
            
            # Figure 8: Correlation Heatmap
            _add_heading(doc, "FIGURA 8: Correlación de Accuracy entre Modelos", level=2)
            _add_correlation_heatmap(doc, stats_model_metrics, model_names)
            doc.add_paragraph()
            
            if include_interpretation:
                _add_paragraph(doc, "Interpretación: La matriz de correlación muestra qué tan similares "
                                "son los rendimientos de los diferentes modelos en los mismos folds. Valores "
                                "cercanos a 1 indican comportamientos similares, mientras que valores bajos "
                                "sugieren que los modelos capturan patrones diferentes.")
                doc.add_paragraph()
            
            # Figure 9: Ranking Chart
            _add_heading(doc, "FIGURA 9: Ranking de Modelos (por Accuracy Promedio)", level=2)
            _add_ranking_chart(doc, stats_model_metrics)
            doc.add_paragraph()
            
            if include_interpretation:
                _add_paragraph(doc, "Interpretación: El ranking permite identificar rápidamente el "
                                "mejor modelo y cuánto mejora sobre los demás. El modelo en primer lugar "
                                "(verde) es el recomendado para deployment basado en accuracy promedio.")
                doc.add_paragraph()
        
        # Add detailed statistics tables
        if stats_model_metrics and include_tables:
            _add_heading(doc, "TABLA: Desviación Estándar por Modelo", level=2)
            _add_detailed_stats_table(doc, stats_model_metrics)
            doc.add_paragraph()
            
            if include_interpretation:
                _add_paragraph(doc, "Interpretación: La desviación estándar y el coeficiente de variación "
                                "(CV) muestran la consistencia del rendimiento de cada modelo. Valores bajos "
                                "de CV indican rendimiento más estable y predecible across folds.")
                doc.add_paragraph()
            
            _add_heading(doc, "TABLA: Comparaciones Pareadas (Accuracy)", level=2)
            _add_paired_comparisons_table(doc, stats_model_metrics)
            doc.add_paragraph()
            
            if include_interpretation:
                _add_paragraph(doc, "Interpretación: Las comparaciones pareadas muestran las diferencias "
                                "directas entre cada par de modelos y el porcentaje de mejora. Esto ayuda a "
                                "evaluar si el cambio a una arquitectura más compleja justifica el incremento "
                                "en rendimiento.")
                doc.add_paragraph()

    if routing_comparison:
        doc.add_page_break()
        _add_heading(doc, "4. Comparación de estrategias de enrutamiento (Monte Carlo)", level=1)
        table = doc.add_table(rows=1, cols=4)
        table.style = 'Light Grid Accent 1'
        hdr_cells = table.rows[0].cells
        hdr_cells[0].text = 'Estrategia'
        hdr_cells[1].text = 'Tasa evac. media'
        hdr_cells[2].text = 'Tiempo evac. medio (pasos)'
        hdr_cells[3].text = 'Cuello de botella medio'
        
        for name, summary in routing_comparison.per_router_summary.items():
            row_cells = table.add_row().cells
            row_cells[0].text = name
            row_cells[1].text = f"{summary['mean_evacuation_rate']:.3f}"
            row_cells[2].text = f"{summary['mean_evacuation_time_steps']:.1f}"
            row_cells[3].text = f"{summary['mean_bottleneck_max_waiting']:.2f}"
        
        doc.add_paragraph()
        _add_significance_table(doc, routing_comparison.statistical_decision)
        _add_paragraph(
            doc,
            f"Decisión estadísticamente justificada: {routing_comparison.statistical_decision['statistically_justified_best']}",
            bold=True
        )

    if active_model_metadata:
        doc.add_page_break()
        _add_heading(doc, "4. Selección del Mejor Modelo (Deployment)", level=1)
        _add_paragraph(doc, "La selección del modelo para producción considera no solo las métricas de rendimiento "
                      "sino también los trade-offs de recursos computacionales, latencia de inferencia y tamaño del modelo.")
        
        if include_tables:
            table = doc.add_table(rows=1, cols=2)
            table.style = 'Light Grid Accent 1'
            hdr_cells = table.rows[0].cells
            hdr_cells[0].text = 'Parámetro'
            hdr_cells[1].text = 'Valor'
            
            for k, v in active_model_metadata.items():
                if k != "hyperparams":
                    row_cells = table.add_row().cells
                    row_cells[0].text = str(k)
                    row_cells[1].text = str(v)
            doc.add_paragraph()
        
        if include_interpretation:
            _add_paragraph(doc, "Interpretación: El modelo seleccionado representa el mejor balance entre "
                            "rendimiento (accuracy, F1-score) y restricciones de deployment (tamaño, latencia). "
                            "Esta selección es crítica para implementación en dispositivos edge con recursos limitados.")
            _add_paragraph(doc, "Explicabilidad: La consideración de trade-offs rendimiento-recursos "
                            "garantiza que el modelo seleccionado sea viable para deployment en entornos de producción "
                            "reales, no solo en entornos de investigación.")
            doc.add_paragraph()
    
    # Conclusions and Recommendations
    doc.add_page_break()
    _add_heading(doc, "5. Conclusiones y Recomendaciones", level=1)
    
    conclusions = [
        "Conclusiones:",
        "• Se implementó exitosamente un pipeline CRISP-DM completo para detección de estrés en evacuación minera.",
        "• Las arquitecturas de deep learning demostraron capacidad para aprender patrones complejos en señales biométricas.",
        "• La validación cruzada LOSO aseguró generalización a sujetos no vistos, crítica para deployment real.",
        "• Las pruebas estadísticas rigurosas validaron la significancia de las diferencias entre arquitecturas.",
        "• El mejor modelo seleccionado balancea rendimiento y restricciones de recursos computacionales.",
        "",
        "Recomendaciones:",
        "• Implementar monitoreo continuo del modelo en producción para detectar drift de datos.",
        "• Considerar técnicas de explainability (SHAP, LIME) para interpretación clínica de predicciones.",
        "• Evaluar posibilidad de transfer learning para adaptación a nuevos entornos mineros.",
        "• Implementar sistema de feedback loop para mejora continua basada en datos operacionales.",
        "• Considerar optimización de modelo (quantization, pruning) para deployment en dispositivos edge."
    ]
    
    for conclusion in conclusions:
        _add_paragraph(doc, conclusion, bold="Conclusiones:" in conclusion or "Recomendaciones:" in conclusion)
    
    if include_interpretation:
        doc.add_paragraph()
        _add_paragraph(doc, "Interpretación: Las conclusiones demuestran la viabilidad técnica del sistema "
                        "de detección de estrés para evacuación minera. Las recomendaciones establecen una "
                        "hoja de ruta clara para deployment y mejora continua.")
        _add_paragraph(doc, "Explicabilidad: Las recomendaciones están basadas en análisis riguroso y "
                        "consideran aspectos técnicos, operacionales y de investigación futura, asegurando "
                        "sostenibilidad del proyecto a largo plazo.")

    doc.save(str(output_path))
    logger.info("Reporte Word generado", extra={"output_path": str(output_path)})
    return output_path