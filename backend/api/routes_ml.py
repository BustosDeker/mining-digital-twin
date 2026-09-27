"""Rutas REST del Motor IA: datasets, EDA, Model Registry, inferencia y
descarga de reportes. El frontend consume esto exclusivamente — nunca
entrena ni accede al sistema de archivos directamente (regla del proyecto).
"""

from __future__ import annotations

from fastapi import APIRouter, HTTPException
from fastapi.responses import FileResponse

from backend.api.schemas import (
    ActivateModelResponse,
    PredictStressRequest,
    PredictStressResponse,
    RunEDARequest,
)
from backend.biometrics.inference import stress_inference_engine
from backend.services import model_registry as mr
from backend.training import synthetic_wesad, wesad_loader  # noqa: F401 - registran los datasets
from backend.training.dataset_loader import DATASET_REGISTRY, get_loader
from backend.training.eda import run_eda
from backend.utils.config import get_settings
from backend.utils.logging_config import get_logger

logger = get_logger(__name__)

router = APIRouter(prefix="/api/ml", tags=["machine_learning"])


# --------------------------------------------------------------------
# Datasets y EDA
# --------------------------------------------------------------------
@router.get("/datasets")
def list_datasets() -> list[dict]:
    result = []
    for name in DATASET_REGISTRY:
        loader = get_loader(name)
        result.append(
            {
                "dataset_name": name,
                "is_available_locally": loader.is_available_locally(),
                "is_synthetic": name != "wesad",
                "n_subjects": len(loader.list_subjects()) if loader.is_available_locally() else 0,
            }
        )
    return result


@router.post("/datasets/{dataset_name}/eda")
def trigger_eda(dataset_name: str, payload: RunEDARequest) -> dict:
    if dataset_name not in DATASET_REGISTRY:
        raise HTTPException(status_code=404, detail=f"Dataset desconocido: {dataset_name}")
    loader = get_loader(dataset_name)
    if not loader.is_available_locally():
        raise HTTPException(
            status_code=409,
            detail=f"'{dataset_name}' no está disponible localmente. Ejecute el script de descarga primero.",
        )
    try:
        summary = run_eda(loader, subject_limit=payload.subject_limit)
    except Exception as exc:  # noqa: BLE001 - se traduce a un error HTTP legible
        raise HTTPException(status_code=500, detail=f"Error generando EDA: {exc}") from exc
    return summary


@router.get("/datasets/{dataset_name}/eda/artifacts")
def list_eda_artifacts(dataset_name: str) -> list[str]:
    settings = get_settings()
    artifacts_dir = settings.ARTIFACTS_DIR / dataset_name / "eda"
    if not artifacts_dir.exists():
        raise HTTPException(status_code=404, detail="Aún no se ha ejecutado EDA para este dataset.")
    return sorted(p.name for p in artifacts_dir.iterdir() if p.is_file())


@router.get("/datasets/{dataset_name}/eda/artifacts/{filename}")
def get_eda_artifact(dataset_name: str, filename: str) -> FileResponse:
    settings = get_settings()
    file_path = settings.ARTIFACTS_DIR / dataset_name / "eda" / filename
    if not file_path.is_file():
        raise HTTPException(status_code=404, detail="Artefacto no encontrado.")
    return FileResponse(file_path)


# --------------------------------------------------------------------
# Model Registry
# --------------------------------------------------------------------
@router.get("/models")
def list_models(architecture_name: str | None = None) -> list[dict]:
    return [m.to_dict() for m in mr.list_models(architecture_name)]


@router.get("/models/active")
def get_active_model() -> dict:
    pointer = mr.get_active_pointer()
    if pointer is None:
        raise HTTPException(status_code=404, detail="No hay ningún modelo activo en producción todavía.")
    metadata = mr.get_model_metadata(pointer["architecture_name"], pointer["version_id"])
    return {"pointer": pointer, "metadata": metadata.to_dict()}


@router.post("/models/{architecture_name}/{version_id}/activate", response_model=ActivateModelResponse)
def activate_model(architecture_name: str, version_id: str) -> ActivateModelResponse:
    try:
        mr.activate_model(architecture_name, version_id)
    except FileNotFoundError as exc:
        raise HTTPException(status_code=404, detail=str(exc)) from exc
    pointer = mr.get_active_pointer()
    return ActivateModelResponse(**pointer)


# --------------------------------------------------------------------
# Inferencia (usa el modelo activo vía el wrapper de biometrics/)
# --------------------------------------------------------------------
@router.post("/predict", response_model=PredictStressResponse)
def predict_stress(payload: PredictStressRequest) -> PredictStressResponse:
    try:
        result = stress_inference_engine.predict_from_features(payload.features)
    except RuntimeError as exc:
        raise HTTPException(status_code=409, detail=str(exc)) from exc
    except NotImplementedError as exc:
        raise HTTPException(status_code=501, detail=str(exc)) from exc
    except ValueError as exc:
        raise HTTPException(status_code=422, detail=str(exc)) from exc

    return PredictStressResponse(
        predicted_class=result.predicted_class,
        probabilities=result.probabilities,
        model_version_id=result.model_version_id,
    )


# --------------------------------------------------------------------
# Reportes PDF
# --------------------------------------------------------------------
@router.get("/reports")
def list_reports() -> list[str]:
    settings = get_settings()
    if not settings.REPORTS_DIR.exists():
        return []
    return sorted(p.name for p in settings.REPORTS_DIR.glob("*.pdf"))


@router.get("/reports/{filename}")
def download_report(filename: str) -> FileResponse:
    settings = get_settings()
    file_path = settings.REPORTS_DIR / filename
    if not file_path.is_file():
        raise HTTPException(status_code=404, detail="Reporte no encontrado.")
    return FileResponse(file_path, media_type="application/pdf", filename=filename)
