"""Descarga reproducible del dataset WESAD, con verificación de checksum.

Uso (en una máquina con acceso a internet completo — este sandbox de
desarrollo NO tiene acceso al dominio de descarga, ver README):

    python -m backend.training.download_wesad

El dataset se coloca en `backend/data/raw/WESAD/`, con la estructura
oficial `WESAD/S2/S2.pkl`, `WESAD/S3/S3.pkl`, etc.

Fuente oficial: Schmidt, P., Reiss, A., Duerichen, R., Marberger, C.,
Van Laerhoven, K. (2018). "Introducing WESAD, a multimodal dataset for
Wearable Stress and Affect Detection." ICMI 2018. UCI Machine Learning
Repository / espejo de la Universidad de Siegen.
"""

from __future__ import annotations

import hashlib
import shutil
import sys
import zipfile
from pathlib import Path

import requests
from tqdm import tqdm

from backend.utils.config import get_settings
from backend.utils.logging_config import get_logger

logger = get_logger(__name__)

_CHUNK_SIZE = 1024 * 1024  # 1 MB


def compute_sha256(file_path: Path) -> str:
    sha256 = hashlib.sha256()
    with open(file_path, "rb") as f:
        for chunk in iter(lambda: f.read(_CHUNK_SIZE), b""):
            sha256.update(chunk)
    return sha256.hexdigest()


def download_file(url: str, destination: Path) -> None:
    destination.parent.mkdir(parents=True, exist_ok=True)
    logger.info("Iniciando descarga de WESAD", extra={"url": url, "destination": str(destination)})

    with requests.get(url, stream=True, timeout=30) as response:
        response.raise_for_status()
        total_size = int(response.headers.get("content-length", 0))
        with open(destination, "wb") as f, tqdm(
            total=total_size, unit="B", unit_scale=True, desc="WESAD.zip"
        ) as progress:
            for chunk in response.iter_content(chunk_size=_CHUNK_SIZE):
                f.write(chunk)
                progress.update(len(chunk))


def verify_checksum(file_path: Path, expected_sha256: str | None) -> bool:
    actual = compute_sha256(file_path)
    if expected_sha256 is None:
        logger.warning(
            "WESAD_EXPECTED_SHA256 no configurado: se omite verificación estricta. "
            "Anote este hash en su .env para verificaciones futuras.",
            extra={"computed_sha256": actual},
        )
        return True
    match = actual.lower() == expected_sha256.lower()
    if not match:
        logger.error(
            "Checksum de WESAD.zip NO coincide con el esperado",
            extra={"expected": expected_sha256, "actual": actual},
        )
    return match


def extract_archive(zip_path: Path, destination_dir: Path) -> None:
    logger.info("Extrayendo WESAD.zip", extra={"destination": str(destination_dir)})
    with zipfile.ZipFile(zip_path, "r") as zf:
        zf.extractall(destination_dir)


def _flatten_if_nested(destination_dir: Path) -> None:
    """Algunos espejos empaquetan el zip con una carpeta raíz adicional
    (p.ej. `WESAD/WESAD/S2/...`); esta función la aplana si existe.
    """
    nested = destination_dir / "WESAD"
    if nested.is_dir() and any(nested.glob("S*/S*.pkl")):
        for item in nested.iterdir():
            shutil.move(str(item), str(destination_dir / item.name))
        nested.rmdir()


def main() -> int:
    settings = get_settings()
    wesad_dir = settings.DATA_RAW_DIR / "WESAD"

    if any(wesad_dir.glob("S*/S*.pkl")):
        logger.info("WESAD ya está presente localmente; nada que hacer.", extra={"path": str(wesad_dir)})
        return 0

    zip_path = settings.DATA_RAW_DIR / "WESAD.zip"
    try:
        if not zip_path.exists():
            download_file(settings.WESAD_URL, zip_path)
    except requests.RequestException as exc:
        logger.error(
            "No se pudo descargar WESAD desde este entorno. Descárguelo manualmente "
            "y coloque el .zip en backend/data/raw/WESAD.zip, o el contenido "
            "extraído directamente en backend/data/raw/WESAD/.",
            extra={"error": str(exc), "url": settings.WESAD_URL},
        )
        return 1

    if not verify_checksum(zip_path, settings.WESAD_EXPECTED_SHA256):
        logger.error("Verificación de checksum fallida. Abortando extracción.")
        return 1

    extract_archive(zip_path, settings.DATA_RAW_DIR)
    _flatten_if_nested(wesad_dir)

    n_subjects = len(list(wesad_dir.glob("S*/S*.pkl")))
    logger.info("WESAD listo para usar", extra={"n_subjects": n_subjects, "path": str(wesad_dir)})
    return 0


if __name__ == "__main__":
    sys.exit(main())
