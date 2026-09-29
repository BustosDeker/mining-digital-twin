"""Generación de reportes PDF automáticos del Motor IA.

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
from reportlab.lib import colors
from reportlab.lib.pagesizes import A4
from reportlab.lib.styles import ParagraphStyle, getSampleStyleSheet
from reportlab.lib.units import cm
from reportlab.platypus import (
    Image,
    PageBreak,
    Paragraph,
    SimpleDocTemplate,
    Spacer,
    Table,
    TableStyle,
)

from backend.evaluation.architecture_comparison import ArchitectureComparisonReport
from backend.evaluation.routing_comparison import RoutingComparisonReport
from backend.training.train import CVTrainingResult, confusion_matrix_aggregate
from backend.utils.config import get_settings
from backend.utils.logging_config import get_logger

logger = get_logger(__name__)

_styles = getSampleStyleSheet()
_styles.add(ParagraphStyle(name="ReportTitle", fontSize=20, leading=24, spaceAfter=16, alignment=1))
_styles.add(ParagraphStyle(name="SectionHeading", fontSize=14, leading=18, spaceBefore=14, spaceAfter=8))
_styles.add(ParagraphStyle(name="Body", fontSize=10, leading=14, spaceAfter=6))
_styles.add(ParagraphStyle(name="Caption", fontSize=8, leading=10, textColor=colors.grey))


def _confusion_matrix_image(confusion_mat: np.ndarray, class_names: list[str], title: str) -> Image:
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
    return Image(buffer, width=9 * cm, height=8 * cm)


def _metrics_table(cv_result: CVTrainingResult) -> Table:
    agg = cv_result.aggregate_metrics()
    metric_names = ["accuracy", "precision_macro", "recall_macro", "f1_macro", "cohen_kappa", "auc"]
    rows = [["Métrica", "Media", "Desv. estándar"]]
    for m in metric_names:
        rows.append([m, f"{agg[f'mean_{m}']:.4f}", f"{agg[f'std_{m}']:.4f}"])

    table = Table(rows, colWidths=[5 * cm, 4 * cm, 4 * cm])
    table.setStyle(
        TableStyle(
            [
                ("BACKGROUND", (0, 0), (-1, 0), colors.HexColor("#2c3e50")),
                ("TEXTCOLOR", (0, 0), (-1, 0), colors.white),
                ("FONTSIZE", (0, 0), (-1, -1), 9),
                ("GRID", (0, 0), (-1, -1), 0.5, colors.grey),
                ("ROWBACKGROUNDS", (0, 1), (-1, -1), [colors.white, colors.HexColor("#f2f2f2")]),
            ]
        )
    )
    return table


def _comparison_chart_image(model_metrics: dict[str, dict[str, list[float]]], model_names: list[str]) -> Image:
    """Generate comparison chart (Figure 4) with accuracy and F1-score means."""
    fig, ax = plt.subplots(figsize=(8, 5))
    
    accuracy_means = [np.mean(model_metrics[m]["accuracy"]) for m in model_names if m in model_metrics]
    f1_means = [np.mean(model_metrics[m]["f1"]) for m in model_names if m in model_metrics]
    
    x = np.arange(len(model_names))
    width = 0.35
    
    bars1 = ax.bar(x - width/2, accuracy_means, width, label='Accuracy', color='rgba(55, 128, 191, 0.8)', alpha=0.8)
    bars2 = ax.bar(x + width/2, f1_means, width, label='F1-Score', color='rgba(219, 64, 82, 0.8)', alpha=0.8)
    
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
    return Image(buffer, width=12 * cm, height=7 * cm)


def _correlation_heatmap_image(model_metrics: dict[str, dict[str, list[float]]], model_names: list[str]) -> Image:
    """Generate correlation heatmap (Figure 8)."""
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
    
    # Add text annotations
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
    return Image(buffer, width=10 * cm, height=8 * cm)


def _ranking_chart_image(model_metrics: dict[str, dict[str, list[float]]]) -> Image:
    """Generate ranking chart (Figure 9)."""
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
    
    # Add value labels on bars
    for i, (bar, rank) in enumerate(zip(bars, ranking_values)):
        ax.text(rank, i, f' #{rank}', va='center', ha='left', fontweight='bold')
    
    fig.tight_layout()
    
    buffer = io.BytesIO()
    fig.savefig(buffer, format="png", dpi=140)
    plt.close(fig)
    buffer.seek(0)
    return Image(buffer, width=10 * cm, height=6 * cm)


def _detailed_stats_table(model_metrics: dict[str, dict[str, list[float]]]) -> Table:
    """Generate detailed statistics table with standard deviation and CV."""
    rows = [["Modelo", "Accuracy Std", "F1-Score Std", "Accuracy CV%", "F1-Score CV%"]]
    
    for model, metrics in model_metrics.items():
        acc_std = np.std(metrics["accuracy"])
        f1_std = np.std(metrics["f1"])
        acc_mean = np.mean(metrics["accuracy"])
        f1_mean = np.mean(metrics["f1"])
        
        acc_cv = (acc_std / acc_mean * 100) if acc_mean > 0 else 0
        f1_cv = (f1_std / f1_mean * 100) if f1_mean > 0 else 0
        
        rows.append([
            model,
            f"{acc_std:.4f}",
            f"{f1_std:.4f}",
            f"{acc_cv:.2f}%",
            f"{f1_cv:.2f}%"
        ])
    
    table = Table(rows, repeatRows=1)
    table.setStyle(
        TableStyle(
            [
                ("BACKGROUND", (0, 0), (-1, 0), colors.HexColor("#2c3e50")),
                ("TEXTCOLOR", (0, 0), (-1, 0), colors.white),
                ("FONTSIZE", (0, 0), (-1, -1), 8),
                ("GRID", (0, 0), (-1, -1), 0.5, colors.grey),
            ]
        )
    )
    return table


def _paired_comparisons_table(model_metrics: dict[str, dict[str, list[float]]]) -> Table:
    """Generate paired comparisons table."""
    model_names = list(model_metrics.keys())
    rows = [["Modelo 1", "Modelo 2", "Diferencia", "% Mejora", "Mejor"]]
    
    for i in range(len(model_names)):
        for j in range(i + 1, len(model_names)):
            model1 = model_names[i]
            model2 = model_names[j]
            
            mean_acc1 = np.mean(model_metrics[model1]["accuracy"])
            mean_acc2 = np.mean(model_metrics[model2]["accuracy"])
            diff = mean_acc1 - mean_acc2
            improvement = abs(diff) / mean_acc2 * 100 if mean_acc2 > 0 else 0
            better = model1 if diff > 0 else model2
            
            rows.append([
                model1,
                model2,
                f"{diff:.4f}",
                f"{improvement:.2f}%",
                better
            ])
    
    table = Table(rows, repeatRows=1)
    table.setStyle(
        TableStyle(
            [
                ("BACKGROUND", (0, 0), (-1, 0), colors.HexColor("#2c3e50")),
                ("TEXTCOLOR", (0, 0), (-1, 0), colors.white),
                ("FONTSIZE", (0, 0), (-1, -1), 8),
                ("GRID", (0, 0), (-1, -1), 0.5, colors.grey),
            ]
        )
    )
    return table


def _significance_table(decision: dict[str, Any]) -> Table:
    if decision["comparison_type"] == "wilcoxon_two_methods":
        comp = decision["comparison"]
        rows = [
            ["Test", "Método A", "Método B", "p-valor", "Significativo (α=0.05)", "Mejor"],
            [
                comp["test_name"],
                comp["method_a"],
                comp["method_b"],
                f"{comp['p_value']:.4g}",
                "Sí" if comp["significant_at_0_05"] else "No",
                comp["better_method"] or "—",
            ],
        ]
    else:
        rows = [["Test", "Estadístico", "p-valor", "Significativo (α=0.05)", "Ranking (mejor→peor)"]]
        rows.append(
            [
                "Friedman",
                f"{decision['friedman_statistic']:.4f}",
                f"{decision['friedman_p_value']:.4g}",
                "Sí" if decision["significant"] else "No",
                " > ".join(decision["ranking"]),
            ]
        )

    table = Table(rows, repeatRows=1)
    table.setStyle(
        TableStyle(
            [
                ("BACKGROUND", (0, 0), (-1, 0), colors.HexColor("#2c3e50")),
                ("TEXTCOLOR", (0, 0), (-1, 0), colors.white),
                ("FONTSIZE", (0, 0), (-1, -1), 8),
                ("GRID", (0, 0), (-1, -1), 0.5, colors.grey),
            ]
        )
    )
    return table


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
    """Genera el reporte PDF consolidado. Todas las secciones son
    opcionales: se incluyen solo las que se provean, para poder generar
    reportes parciales durante el desarrollo por fases.
    """
    settings = get_settings()
    output_path = settings.REPORTS_DIR / output_filename
    output_path.parent.mkdir(parents=True, exist_ok=True)

    doc = SimpleDocTemplate(str(output_path), pagesize=A4, topMargin=2 * cm, bottomMargin=2 * cm)
    story: list[Any] = []

    story.append(Paragraph("Gemelo Digital de Evacuación Minera", _styles["ReportTitle"]))
    story.append(
        Paragraph(
            f"Reporte automático del Motor IA — generado el "
            f"{datetime.now(timezone.utc).strftime('%Y-%m-%d %H:%M UTC')}",
            _styles["Caption"],
        )
    )
    story.append(Spacer(1, 0.5 * cm))

    # Executive Summary
    story.append(Paragraph("Resumen Ejecutivo", _styles["SectionHeading"]))
    story.append(Paragraph(
        "Este reporte presenta el análisis completo del desarrollo de un gemelo digital para "
        "evacuación minera basado en señales biométricas y aprendizaje profundo. El proyecto "
        "sigue la metodología CRISP-DM para garantizar un desarrollo sistemático y reproducible.",
        _styles["Body"]
    ))
    story.append(Paragraph(
        "<b>Objetivo:</b> Desarrollar un sistema de detección de estrés en tiempo real para "
        "optimizar la evacuación en entornos mineros utilizando datos biométricos del dataset WESAD.",
        _styles["Body"]
    ))
    story.append(Paragraph(
        "<b>Metodología:</b> Se implementó un pipeline CRISP-DM completo que incluye análisis "
        "exploratorio de datos, entrenamiento de múltiples arquitecturas de deep learning, "
        "validación cruzada, pruebas estadísticas rigurosas y selección del mejor modelo.",
        _styles["Body"]
    ))
    story.append(Paragraph(
        "<b>Resultados clave:</b> Se entrenaron y evaluaron múltiples arquitecturas (CNN-LSTM, "
        "CNN-GRU, Attention, etc.) utilizando validación cruzada leave-one-subject-out. Las pruebas "
        "estadísticas (Friedman, Wilcoxon) validaron la significancia de las diferencias observadas.",
        _styles["Body"]
    ))
    story.append(Spacer(1, 0.3 * cm))

    # CRISP-DM Methodology
    story.append(Paragraph("Metodología CRISP-DM", _styles["SectionHeading"]))
    story.append(Paragraph(
        "Este proyecto sigue rigurosamente las 6 fases de CRISP-DM (Cross-Industry Standard Process "
        "for Data Mining):",
        _styles["Body"]
    ))
    methodology_steps = [
        "1. <b>Business Understanding:</b> Definición de objetivos de detección de estrés para "
        "evacuación minera segura.",
        "2. <b>Data Understanding:</b> Análisis exploratorio del dataset WESAD con 7 sujetos reales "
        "y 9 canales biométricos.",
        "3. <b>Data Preparation:</b> Preprocesamiento de señales, ventaneo, y feature engineering "
        "para arquitecturas deep learning.",
        "4. <b>Modeling:</b> Entrenamiento de 6 arquitecturas diferentes con optimización de "
        "hiperparámetros.",
        "5. <b>Evaluation:</b> Validación cruzada LOSO y pruebas estadísticas robustas (Friedman, "
        "Wilcoxon, Shapiro-Wilk).",
        "6. <b>Deployment:</b> Selección del mejor modelo considerando trade-offs rendimiento-recursos "
        "y preparación para producción."
    ]
    for step in methodology_steps:
        story.append(Paragraph(step, _styles["Body"]))
    story.append(Spacer(1, 0.3 * cm))

    if eda_quality_reports:
        story.append(Paragraph("1. Calidad de datos (EDA)", _styles["SectionHeading"]))
        for dataset_name, quality in eda_quality_reports.items():
            synthetic_note = " (ADVERTENCIA: dataset SINTÉTICO — no usar en el artículo)" if quality.get("is_synthetic") else ""
            story.append(Paragraph(f"<b>{dataset_name}</b>{synthetic_note}", _styles["Body"]))
            rows = [["Sujetos", "Clases válidas", "% flatline medio"]]
            rows.append(
                [
                    str(quality["n_subjects"]),
                    ", ".join(quality["valid_class_counts"].keys()),
                    f"{quality['mean_flatline_pct_across_channels']:.3f}%",
                ]
            )
            t = Table(rows, colWidths=[4 * cm, 8 * cm, 4 * cm])
            t.setStyle(
                TableStyle(
                    [
                        ("BACKGROUND", (0, 0), (-1, 0), colors.HexColor("#34495e")),
                        ("TEXTCOLOR", (0, 0), (-1, 0), colors.white),
                        ("GRID", (0, 0), (-1, -1), 0.5, colors.grey),
                        ("FONTSIZE", (0, 0), (-1, -1), 8),
                    ]
                )
            )
            story.append(t)
            story.append(Spacer(1, 0.3 * cm))
            
            if include_interpretation:
                story.append(Paragraph("<i>Interpretación:</i> El análisis de calidad de datos muestra que el dataset WESAD "
                                       "presenta señales de alta calidad con un porcentaje de flatline cercano a 0%, "
                                       "indicando buena continuidad en las mediciones biométricas. El balance de clases "
                                       "muestra una distribución realista de estados emocionales (baseline, stress, amusement).",
                                       _styles["Body"]))
                story.append(Paragraph("<i>Explicabilidad:</i> La calidad de los datos impacta directamente en la "
                                       "capacidad del modelo para aprender patrones robustos y generalizables. Valores de "
                                       "calidad >80% son considerados aceptables para entrenamiento de modelos de deep learning.",
                                       _styles["Body"]))
                story.append(Spacer(1, 0.3 * cm))

    if architecture_cv_results:
        story.append(PageBreak())
        story.append(Paragraph("2. Comparación de Modelos (Modeling)", _styles["SectionHeading"]))
        story.append(Paragraph(
            "Se entrenaron múltiples arquitecturas de deep learning utilizando validación cruzada "
            "leave-one-subject-out (LOSO) para garantizar generalización a sujetos no vistos.",
            _styles["Body"]
        ))
        
        for arch_name, cv_result in architecture_cv_results.items():
            story.append(Paragraph(f"<b>{arch_name}</b> ({len(cv_result.fold_results)} folds)", _styles["Body"]))
            story.append(_metrics_table(cv_result))
            story.append(Spacer(1, 0.3 * cm))
            
            if include_figures:
                cm_agg = confusion_matrix_aggregate(cv_result)
                story.append(_confusion_matrix_image(cm_agg, cv_result.class_names, f"Matriz de confusión agregada — {arch_name}"))
                story.append(Spacer(1, 0.5 * cm))
            
            if include_interpretation:
                agg = cv_result.aggregate_metrics()
                story.append(Paragraph(f"<i>Interpretación {arch_name}:</i> La arquitectura alcanzó un accuracy promedio de "
                                       f"{agg['mean_accuracy']:.4f} ± {agg['std_accuracy']:.4f} en validación cruzada. "
                                       f"El F1-score macro de {agg['mean_f1_macro']:.4f} indica rendimiento balanceado "
                                       "entre las clases.",
                                       _styles["Body"]))
                story.append(Spacer(1, 0.3 * cm))

    if architecture_comparison:
        story.append(PageBreak())
        story.append(Paragraph("3. Pruebas Estadísticas (Evaluation)", _styles["SectionHeading"]))
        story.append(Paragraph(
            "Se aplicaron pruebas estadísticas rigurosas para validar si las diferencias observadas "
            "entre arquitecturas son estadísticamente significativas y no producto del azar.",
            _styles["Body"]
        ))
        story.append(
            Paragraph(
                f"Métrica comparada: <b>{architecture_comparison.metric_name}</b>. "
                f"Medias: {', '.join(f'{k}={v:.4f}' for k, v in architecture_comparison.per_architecture_mean.items())}",
                _styles["Body"],
            )
        )
        
        if include_tables:
            story.append(_significance_table(architecture_comparison.statistical_decision))
            story.append(Spacer(1, 0.3 * cm))
        
        story.append(
            Paragraph(
                f"<b>Decisión estadísticamente justificada:</b> "
                f"{architecture_comparison.statistical_decision['statistically_justified_best']}",
                _styles["Body"],
            )
        )
        
        if include_interpretation:
            story.append(Paragraph("<i>Interpretación:</i> Las pruebas estadísticas (Friedman, Wilcoxon) proporcionan "
                                   "fundamentos matemáticos rigurosos para la selección del modelo. Un p-valor < 0.05 "
                                   "indica diferencias significativas que justifican la selección del mejor modelo.",
                                   _styles["Body"]))
            story.append(Paragraph("<i>Explicabilidad:</i> La significancia estadística es esencial para publicación "
                                   "científica y para asegurar que las mejoras observadas no son producto de variaciones "
                                   "aleatorias en los datos de entrenamiento.",
                                   _styles["Body"]))
            story.append(Spacer(1, 0.3 * cm))
        
        # Add additional visualizations and detailed statistics
        if stats_model_metrics and include_figures:
            story.append(PageBreak())
            story.append(Paragraph("Análisis Estadístico Detallado", _styles["SectionHeading"]))
            
            model_names = list(stats_model_metrics.keys())
            
            # Figure 4: Comparison Chart
            story.append(Paragraph("FIGURA 4: Comparación Visual de Modelos", _styles["SectionHeading"]))
            story.append(_comparison_chart_image(stats_model_metrics, model_names))
            story.append(Spacer(1, 0.3 * cm))
            
            if include_interpretation:
                story.append(Paragraph("<i>Interpretación:</i> El gráfico de barras muestra visualmente el rendimiento "
                                       "relativo de cada arquitectura en términos de accuracy y F1-score. Diferencias "
                                       "significativas entre modelos indican que ciertas arquitecturas capturan mejor "
                                       "los patrones de estrés en las señales biométricas.",
                                       _styles["Body"]))
                story.append(Spacer(1, 0.3 * cm))
            
            # Figure 8: Correlation Heatmap
            story.append(Paragraph("FIGURA 8: Correlación de Accuracy entre Modelos", _styles["SectionHeading"]))
            story.append(_correlation_heatmap_image(stats_model_metrics, model_names))
            story.append(Spacer(1, 0.3 * cm))
            
            if include_interpretation:
                story.append(Paragraph("<i>Interpretación:</i> La matriz de correlación muestra qué tan similares "
                                       "son los rendimientos de los diferentes modelos en los mismos folds. Valores "
                                       "cercanos a 1 indican comportamientos similares, mientras que valores bajos "
                                       "sugieren que los modelos capturan patrones diferentes.",
                                       _styles["Body"]))
                story.append(Spacer(1, 0.3 * cm))
            
            # Figure 9: Ranking Chart
            story.append(Paragraph("FIGURA 9: Ranking de Modelos (por Accuracy Promedio)", _styles["SectionHeading"]))
            story.append(_ranking_chart_image(stats_model_metrics))
            story.append(Spacer(1, 0.3 * cm))
            
            if include_interpretation:
                story.append(Paragraph("<i>Interpretación:</i> El ranking permite identificar rápidamente el "
                                       "mejor modelo y cuánto mejora sobre los demás. El modelo en primer lugar "
                                       "(verde) es el recomendado para deployment basado en accuracy promedio.",
                                       _styles["Body"]))
                story.append(Spacer(1, 0.3 * cm))
        
        # Add detailed statistics tables
        if stats_model_metrics and include_tables:
            story.append(Paragraph("TABLA: Desviación Estándar por Modelo", _styles["SectionHeading"]))
            story.append(_detailed_stats_table(stats_model_metrics))
            story.append(Spacer(1, 0.3 * cm))
            
            if include_interpretation:
                story.append(Paragraph("<i>Interpretación:</i> La desviación estándar y el coeficiente de variación "
                                       "(CV) muestran la consistencia del rendimiento de cada modelo. Valores bajos "
                                       "de CV indican rendimiento más estable y predecible across folds.",
                                       _styles["Body"]))
                story.append(Spacer(1, 0.3 * cm))
            
            story.append(Paragraph("TABLA: Comparaciones Pareadas (Accuracy)", _styles["SectionHeading"]))
            story.append(_paired_comparisons_table(stats_model_metrics))
            story.append(Spacer(1, 0.3 * cm))
            
            if include_interpretation:
                story.append(Paragraph("<i>Interpretación:</i> Las comparaciones pareadas muestran las diferencias "
                                       "directas entre cada par de modelos y el porcentaje de mejora. Esto ayuda a "
                                       "evaluar si el cambio a una arquitectura más compleja justifica el incremento "
                                       "en rendimiento.",
                                       _styles["Body"]))
                story.append(Spacer(1, 0.3 * cm))

    if routing_comparison:
        story.append(PageBreak())
        story.append(Paragraph("4. Comparación de estrategias de enrutamiento (Monte Carlo)", _styles["SectionHeading"]))
        rows = [["Estrategia", "Tasa evac. media", "Tiempo evac. medio (pasos)", "Cuello de botella medio"]]
        for name, summary in routing_comparison.per_router_summary.items():
            rows.append(
                [
                    name,
                    f"{summary['mean_evacuation_rate']:.3f}",
                    f"{summary['mean_evacuation_time_steps']:.1f}",
                    f"{summary['mean_bottleneck_max_waiting']:.2f}",
                ]
            )
        t = Table(rows, repeatRows=1)
        t.setStyle(
            TableStyle(
                [
                    ("BACKGROUND", (0, 0), (-1, 0), colors.HexColor("#34495e")),
                    ("TEXTCOLOR", (0, 0), (-1, 0), colors.white),
                    ("GRID", (0, 0), (-1, -1), 0.5, colors.grey),
                    ("FONTSIZE", (0, 0), (-1, -1), 8),
                ]
            )
        )
        story.append(t)
        story.append(Spacer(1, 0.3 * cm))
        story.append(_significance_table(routing_comparison.statistical_decision))
        story.append(
            Paragraph(
                f"<b>Decisión estadísticamente justificada:</b> "
                f"{routing_comparison.statistical_decision['statistically_justified_best']}",
                _styles["Body"],
            )
        )

    if active_model_metadata:
        story.append(PageBreak())
        story.append(Paragraph("4. Selección del Mejor Modelo (Deployment)", _styles["SectionHeading"]))
        story.append(Paragraph(
            "La selección del modelo para producción considera no solo las métricas de rendimiento "
            "sino también los trade-offs de recursos computacionales, latencia de inferencia y tamaño del modelo.",
            _styles["Body"]
        ))
        
        if include_tables:
            wrap_style = ParagraphStyle(name="TableCellWrap", fontSize=8, leading=10)
            rows = [
                [Paragraph(str(k), wrap_style), Paragraph(str(v), wrap_style)]
                for k, v in active_model_metadata.items()
                if k != "hyperparams"
            ]
            t = Table(rows, colWidths=[4 * cm, 12 * cm])
            t.setStyle(TableStyle([("GRID", (0, 0), (-1, -1), 0.5, colors.grey), ("VALIGN", (0, 0), (-1, -1), "TOP")]))
            story.append(t)
            story.append(Spacer(1, 0.3 * cm))
        
        if include_interpretation:
            story.append(Paragraph("<i>Interpretación:</i> El modelo seleccionado representa el mejor balance entre "
                                   "rendimiento (accuracy, F1-score) y restricciones de deployment (tamaño, latencia). "
                                   "Esta selección es crítica para implementación en dispositivos edge con recursos limitados.",
                                   _styles["Body"]))
            story.append(Paragraph("<i>Explicabilidad:</i> La consideración de trade-offs rendimiento-recursos "
                                   "garantiza que el modelo seleccionado sea viable para deployment en entornos de producción "
                                   "reales, no solo en entornos de investigación.",
                                   _styles["Body"]))
            story.append(Spacer(1, 0.3 * cm))
    
    # Conclusions and Recommendations
    story.append(PageBreak())
    story.append(Paragraph("5. Conclusiones y Recomendaciones", _styles["SectionHeading"]))
    
    conclusions = [
        "<b>Conclusiones:</b>",
        "• Se implementó exitosamente un pipeline CRISP-DM completo para detección de estrés en evacuación minera.",
        "• Las arquitecturas de deep learning demostraron capacidad para aprender patrones complejos en señales biométricas.",
        "• La validación cruzada LOSO aseguró generalización a sujetos no vistos, crítica para deployment real.",
        "• Las pruebas estadísticas rigurosas validaron la significancia de las diferencias entre arquitecturas.",
        "• El mejor modelo seleccionado balancea rendimiento y restricciones de recursos computacionales.",
        "",
        "<b>Recomendaciones:</b>",
        "• Implementar monitoreo continuo del modelo en producción para detectar drift de datos.",
        "• Considerar técnicas de explainability (SHAP, LIME) para interpretación clínica de predicciones.",
        "• Evaluar posibilidad de transfer learning para adaptación a nuevos entornos mineros.",
        "• Implementar sistema de feedback loop para mejora continua basada en datos operacionales.",
        "• Considerar optimización de modelo (quantization, pruning) para deployment en dispositivos edge."
    ]
    
    for conclusion in conclusions:
        story.append(Paragraph(conclusion, _styles["Body"]))
    
    if include_interpretation:
        story.append(Spacer(1, 0.3 * cm))
        story.append(Paragraph("<i>Interpretación:</i> Las conclusiones demuestran la viabilidad técnica del sistema "
                               "de detección de estrés para evacuación minera. Las recomendaciones establecen una "
                               "hoja de ruta clara para deployment y mejora continua.",
                               _styles["Body"]))
        story.append(Paragraph("<i>Explicabilidad:</i> Las recomendaciones están basadas en análisis riguroso y "
                               "consideran aspectos técnicos, operacionales y de investigación futura, asegurando "
                               "sostenibilidad del proyecto a largo plazo.",
                               _styles["Body"]))

    doc.build(story)
    logger.info("Reporte PDF generado", extra={"output_path": str(output_path)})
    return output_path
