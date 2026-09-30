"""Punto de entrada: asyncio.run, carga de .env/config y bucle de chat."""

from __future__ import annotations

import asyncio
import contextlib
import os
import sys
from typing import Any
from uuid import uuid4

from dotenv import load_dotenv
from langchain_core.messages import AIMessage, HumanMessage, ToolMessage
from langgraph.checkpoint.sqlite.aio import AsyncSqliteSaver

from src.graph import construir_grafo
from src.providers.gemini import GeminiClient
from src.schema.models import BASE_DIR, ConfigParams, cargar_configuracion
from src.state import texto_plano
from src.trace import Traza

SALIDAS = {"salir", "exit", "q"}


def traducir_eventos(nodo: str, datos: dict[str, Any]) -> list[dict[str, Any]]:
    """Convierte una actualización del grafo en eventos de la traza JSON Lines."""
    if not datos:
        return []

    eventos: list[dict[str, Any]] = []

    if nodo == "supervisor":
        eventos.append(
            {
                "tipo": "pensamiento",
                "nodo": "supervisor",
                "decision": datos.get("instruccion_actual"),
                "ruta": datos.get("next_agent"),
                "intentos_correccion": datos.get("intentos_correccion"),
            }
        )
        return eventos

    if nodo in ("investigador", "analista"):
        for contribucion in datos.get("contribuciones") or []:
            metadata = contribucion.get("metadata") or {}
            eventos.append(
                {
                    "tipo": "contribucion",
                    "nodo": contribucion.get("agente", nodo),
                    "clase": contribucion.get("tipo"),
                    "contenido": contribucion.get("contenido"),
                    "error": metadata.get("error"),
                }
            )
            for accion in metadata.get("herramientas") or []:
                eventos.append(
                    {
                        "tipo": "accion",
                        "nodo": contribucion.get("agente", nodo),
                        "herramienta": accion.get("herramienta"),
                        "argumentos": accion.get("argumentos"),
                    }
                )
            for observacion in metadata.get("observaciones") or []:
                eventos.append(
                    {
                        "tipo": "observacion",
                        "nodo": "tools",
                        "herramienta": observacion.get("herramienta"),
                        "resultado": observacion.get("resultado"),
                    }
                )
        return eventos

    if nodo == "sintesis":
        for mensaje in datos.get("messages") or []:
            if isinstance(mensaje, AIMessage):
                eventos.append(
                    {
                        "tipo": "respuesta_final",
                        "nodo": "sintesis",
                        "contenido": mensaje.content,
                        "texto": texto_plano(mensaje.content),
                    }
                )
        return eventos

    if nodo == "tools":
        for mensaje in datos.get("messages") or []:
            if isinstance(mensaje, ToolMessage):
                eventos.append(
                    {
                        "tipo": "observacion",
                        "nodo": "tools",
                        "herramienta": mensaje.name,
                        "resultado": mensaje.content,
                    }
                )
        return eventos

    return eventos


def _contenido(mensaje: Any) -> str:
    return texto_plano(getattr(mensaje, "content", ""))


async def _preguntar_traza() -> bool:
    """Pregunta si quiere verse la traza en pantalla; por defecto es No."""
    try:
        respuesta = await asyncio.to_thread(input, "¿Ver la traza en pantalla? [s/N] ")
    except (EOFError, KeyboardInterrupt):
        print()
        return False
    return respuesta.strip().casefold() in {"s", "si", "sí", "y", "yes"}


async def correr(cfg: ConfigParams) -> int:
    cliente = GeminiClient(cfg)

    async with AsyncSqliteSaver.from_conn_string(cfg.sqlite_db_path) as saver:
        grafo = construir_grafo(cliente, cfg, saver)

        thread_id = f"sesion-{uuid4().hex[:12]}"
        config: dict[str, Any] = {
            "configurable": {"thread_id": thread_id},
            "recursion_limit": cfg.recursion_limit,
        }

        traza = Traza(
            cfg.trace_dir,
            thread_id,
            {
                "model_name": cfg.model_name,
                "temperature": cfg.temperature,
                "recursion_limit": cfg.recursion_limit,
                "max_iterations_agentes": cfg.max_iterations_agentes,
                "max_intentos_correccion": cfg.max_intentos_correccion,
            },
        )
        await traza.registrar_sesion()

        print(f"[sesión] thread_id={thread_id}")
        print(
            f"[config] modelo={cfg.model_name} | iteraciones por agente="
            f"{cfg.max_iterations_agentes} | intentos máx={cfg.max_intentos_correccion}\n"
        )
        print("Escribí tu pregunta. 'salir', 'exit' o 'q' cierran la sesión.\n")

        turno = 0

        while True:
            try:
                entrada = await asyncio.to_thread(input, "Tú: ")
            except (EOFError, KeyboardInterrupt):
                print()
                break

            entrada = entrada.strip()
            if not entrada:
                continue
            if entrada.lower() in SALIDAS:
                if await _preguntar_traza():
                    try:
                        await traza.mostrar_en_pantalla()
                    except (OSError, ValueError, KeyError, IndexError):
                        print("[aviso] No se pudo mostrar la traza.")
                break

            turno += 1
            await traza.registrar({"tipo": "entrada_usuario", "contenido": entrada, "turno": turno})

            try:
                async for updates in grafo.astream(
                    {
                        "messages": [HumanMessage(entrada)],
                        "intentos_correccion": 0,
                    },
                    config,
                    stream_mode="updates",
                ):
                    for nodo, datos in updates.items():
                        for evento in traducir_eventos(nodo, datos):
                            evento["turno"] = turno
                            await traza.registrar(evento)
                            if evento["tipo"] == "pensamiento":
                                print(
                                    f"\n[supervisor] -> {evento.get('ruta')}: "
                                    f"{evento.get('decision')}"
                                )
            except Exception as exc:
                await traza.registrar({"tipo": "error", "mensaje": str(exc), "turno": turno})
                print(f"[error] {type(exc).__name__}: {exc}\n")
                continue

            estado = await grafo.aget_state(config)
            mensajes = list(estado.values.get("messages") or [])
            ultimo = mensajes[-1] if mensajes else None
            respuesta = _contenido(ultimo) if isinstance(ultimo, AIMessage) else ""
            print(f"\nAgente: {respuesta}\n")

    return 0


def _configurar_consola() -> None:
    """Evita que un carácter fuera de la codepage corte la sesión (Windows)."""
    for flujo in (sys.stdout, sys.stderr):
        reconfigurar = getattr(flujo, "reconfigure", None)
        if reconfigurar is None:
            continue
        with contextlib.suppress(OSError, ValueError):
            reconfigurar(errors="replace")


def main() -> int:
    _configurar_consola()
    load_dotenv(BASE_DIR / "config" / ".env")

    try:
        cfg: ConfigParams = cargar_configuracion()
    except (OSError, ValueError) as exc:
        print(f"[ERROR] No se pudo cargar la configuración: {exc}")
        return 1

    if not os.environ.get("GEMINI_API_KEY", "").strip():
        print("[ERROR] Falta GEMINI_API_KEY: completa config/.env con tu key de Gemini.")
        return 1

    if not os.environ.get("PINECONE_API_KEY", "").strip():
        print(
            "[aviso] Falta PINECONE_API_KEY: la herramienta buscar_informacion "
            "no podrá consultar la base de conocimiento."
        )

    try:
        return asyncio.run(correr(cfg))
    except KeyboardInterrupt:
        print("\n[sesión] interrumpida")
        return 0


if __name__ == "__main__":
    raise SystemExit(main())
