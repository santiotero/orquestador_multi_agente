"""Esquemas Pydantic del sistema (SupervisorOutput, ConfigParams) y carga de config."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Literal

from pydantic import BaseModel, Field

BASE_DIR = Path(__file__).resolve().parents[2]
CONFIG_PATH = BASE_DIR / "config" / "config.json"

_CLAVES_DE_RUTA = ("sqlite_db_path", "trace_dir", "data_dir", "bm25_path")


class SupervisorOutput(BaseModel):
    """Decisión estructurada que devuelve el supervisor en cada paso."""

    next: Literal["investigador", "analista", "sintesis"] = Field(
        description="El siguiente agente a ejecutar o 'sintesis' si el trabajo terminó."
    )
    instruccion: str = Field(
        description=(
            "Instrucción ultra-específica de lo que debe hacer el siguiente agente en este turno."
        )
    )
    razon: str = Field(description="Justificación interna del supervisor para tomar esta decisión.")


class ConfigParams(BaseModel):
    """Configuración central del orquestador (config/config.json)."""

    model_name: str = "gemini-3.1-flash-lite"
    temperature: float = 0.0
    max_tokens: int = 512
    recursion_limit: int = 15
    max_iterations_agentes: int = 3
    max_tokens_contexto: int = 6000
    max_intentos_correccion: int = 3
    sqlite_db_path: str = "orquestador.db"
    trace_dir: str = "trace"
    data_dir: str = "ingest/data"
    bm25_path: str = "src/vectorstore/bm25_index.pickle"
    embedding_model: str = "models/gemini-embedding-001"
    embedding_dim: int = 1536
    chunk_size: int = 600
    chunk_overlap: int = 100
    top_k: int = 5
    score_threshold: float = 0.75
    index_name: str = "cloudindex"
    namespace: str = "multiagente"

    def ruta(self, clave: str) -> Path:
        """Devuelve una ruta de configuración ya resuelta contra la raíz del proyecto."""
        valor = Path(getattr(self, clave))
        return valor if valor.is_absolute() else (BASE_DIR / valor).resolve()


def cargar_configuracion(ruta: str | Path | None = None) -> ConfigParams:
    """Lee config/config.json y devuelve un ConfigParams con las rutas resueltas.

    Las rutas relativas (`sqlite_db_path`, `trace_dir`, `data_dir`, `bm25_path`)
    se resuelven contra la raíz del proyecto para que el programa funcione desde
    cualquier directorio de trabajo.
    """
    path = Path(ruta) if ruta is not None else CONFIG_PATH
    if not path.is_absolute():
        path = BASE_DIR / path
    with open(path, encoding="utf-8") as fh:
        crudos = json.load(fh)
    config = ConfigParams(**crudos)
    for clave in _CLAVES_DE_RUTA:
        setattr(config, clave, str(config.ruta(clave)))
    return config
