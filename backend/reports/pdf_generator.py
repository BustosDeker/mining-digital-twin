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

    if architecture_cv_results:
        story.append(PageBreak())
        story.append(Paragraph("2. Métricas por arquitectura (validación cruzada)", _styles["SectionHeading"]))
        for arch_name, cv_result in architecture_cv_results.items():
            story.append(Paragraph(f"<b>{arch_name}</b> ({len(cv_result.fold_results)} folds)", _styles["Body"]))
            story.append(_metrics_table(cv_result))
            story.append(Spacer(1, 0.3 * cm))
            cm_agg = confusion_matrix_aggregate(cv_result)
            story.append(_confusion_matrix_image(cm_agg, cv_result.class_names, f"Matriz de confusión agregada — {arch_name}"))
            story.append(Spacer(1, 0.5 * cm))

    if architecture_comparison:
        story.append(PageBreak())
        story.append(Paragraph("3. Comparación estadística de arquitecturas", _styles["SectionHeading"]))
        story.append(
            Paragraph(
                f"Métrica comparada: <b>{architecture_comparison.metric_name}</b>. "
                f"Medias: {', '.join(f'{k}={v:.4f}' for k, v in architecture_comparison.per_architecture_mean.items())}",
                _styles["Body"],
            )
        )
        story.append(_significance_table(architecture_comparison.statistical_decision))
        story.append(
            Paragraph(
                f"<b>Decisión estadísticamente justificada:</b> "
                f"{architecture_comparison.statistical_decision['statistically_justified_best']}",
                _styles["Body"],
            )
        )

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
        story.append(Paragraph("5. Modelo activo en producción", _styles["SectionHeading"]))
        wrap_style = ParagraphStyle(name="TableCellWrap", fontSize=8, leading=10)
        rows = [
            [Paragraph(str(k), wrap_style), Paragraph(str(v), wrap_style)]
            for k, v in active_model_metadata.items()
            if k != "hyperparams"
        ]
        t = Table(rows, colWidths=[4 * cm, 12 * cm])
        t.setStyle(TableStyle([("GRID", (0, 0), (-1, -1), 0.5, colors.grey), ("VALIGN", (0, 0), (-1, -1), "TOP")]))
        story.append(t)

    doc.build(story)
    logger.info("Reporte PDF generado", extra={"output_path": str(output_path)})
    return output_path
