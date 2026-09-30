"""Vaciado de la base del proyecto: namespace de Pinecone + índice BM25.

Comando:
    poetry run python -m ingest.delete          pide confirmación
    poetry run python -m ingest.delete --yes    sin confirmación

Borra el namespace configurado (`multiagente`) del índice `cloudindex` y el
pickle de BM25, dejando el proyecto en condiciones de volver a ingestar.
"""

from __future__ import annotations

import logging
import os
import sys

from dotenv import load_dotenv

from src.schema.models import BASE_DIR, ConfigParams, cargar_configuracion
from src.vectorstore.rag_system import RAGSystem

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger("orquestador.delete")


def confirmar(mensaje: str) -> bool:
    """Pide confirmación antes de borrar; `--yes` o `-y` la salta."""
    if any(flag in sys.argv for flag in ("--yes", "-y")):
        return True
    try:
        respuesta = input(f"{mensaje} [s/N] ").strip().casefold()
    except (EOFError, KeyboardInterrupt):
        print()
        return False
    return respuesta in {"s", "si", "sí", "y", "yes"}


def main() -> None:
    """Vacía el namespace de Pinecone y elimina el pickle de BM25."""
    logger.info("=" * 50)
    logger.info("Vaciado de Pinecone + BM25 - Orquestador Multi-Agente")
    logger.info("=" * 50)

    load_dotenv(BASE_DIR / "config" / ".env")
    cfg: ConfigParams = cargar_configuracion()

    pinecone_api_key = os.environ.get("PINECONE_API_KEY", "").strip()
    if not pinecone_api_key:
        logger.error("PINECONE_API_KEY no configurada en config/.env")
        raise SystemExit(1)

    logger.info("Índice: %s | Namespace: %s", cfg.index_name, cfg.namespace)
    logger.info("BM25: %s", cfg.bm25_path)

    if not confirmar("¿Vaciar el namespace y borrar el índice BM25?"):
        logger.info("Operación cancelada: no se modificó nada.")
        raise SystemExit(0)

    rag = RAGSystem(
        cfg,
        pinecone_api_key,
        os.environ.get("GEMINI_API_KEY", "").strip() or None,
    )
    rag.delete_all()

    try:
        stats = rag.index.describe_index_stats()
    except Exception as exc:
        logger.error("No se pudo verificar Pinecone tras el borrado: %s", exc)
        raise SystemExit(1) from exc

    namespace = stats.namespaces.get(cfg.namespace)
    pinecone_limpio = namespace is None or namespace.vector_count == 0
    pickle_limpio = not rag.has_bm25()

    if pinecone_limpio and pickle_limpio:
        logger.info(
            "Listo: namespace '%s' vacío y BM25 borrado. "
            "La ingesta puede volver a correr con `python -m ingest.ingest`.",
            cfg.namespace,
        )
        raise SystemExit(0)

    logger.error(
        "Borrado incompleto -> namespace '%s' con vectores: %s | BM25 presente: %s",
        cfg.namespace,
        not pinecone_limpio,
        not pickle_limpio,
    )
    raise SystemExit(1)


if __name__ == "__main__":
    main()
