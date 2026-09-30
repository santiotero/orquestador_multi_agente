"""Trazas de observabilidad en formato JSON Lines (.jsonl)."""

from __future__ import annotations

import json
import textwrap
from datetime import datetime
from pathlib import Path
from typing import Any

import aiofiles


class Traza:
    """Persiste la traza de una sesión en `trace/trace_{thread_id}.jsonl`.

    Cada evento es una línea JSON independiente añadida al final del archivo
    (JSON Lines): append nativo, sin reescribir el documento completo, más
    rápido y sin bloquear el event loop (escritura con `aiofiles`).
    """

    def __init__(self, trace_dir: str, thread_id: str, parametros: dict[str, Any]) -> None:
        self.thread_id = thread_id
        self.ruta = Path(trace_dir) / f"trace_{thread_id}.jsonl"
        self._parametros = parametros
        self._contador: int | None = None

    async def _numerador(self) -> int:
        if self._contador is not None:
            return self._contador
        if self.ruta.exists():
            async with aiofiles.open(self.ruta, encoding="utf-8") as fh:
                lineas = [linea for linea in (await fh.read()).splitlines() if linea.strip()]
            self._contador = len(lineas)
        else:
            self._contador = 0
        return self._contador

    async def registrar(self, evento: dict[str, Any]) -> dict[str, Any]:
        """Añade una línea JSON al final del archivo .jsonl y devuelve el evento."""
        numero = (await self._numerador()) + 1
        self._contador = numero

        entrada = {
            "n": numero,
            "ts": datetime.now().isoformat(timespec="seconds"),
            **evento,
        }

        self.ruta.parent.mkdir(parents=True, exist_ok=True)
        async with aiofiles.open(self.ruta, "a", encoding="utf-8") as fh:
            await fh.write(json.dumps(entrada, ensure_ascii=False, default=str) + "\n")
        return entrada

    async def registrar_sesion(self) -> dict[str, Any]:
        """Primer evento del archivo: identifica la sesión y sus parámetros."""
        return await self.registrar(
            {
                "tipo": "sesion",
                "thread_id": self.thread_id,
                "parametros": self._parametros,
            }
        )

    async def mostrar_en_pantalla(self) -> None:
        """Lee el `.jsonl` de esta sesión y lo imprime como texto legible.

        Sólo lectura: el archivo de la traza no se modifica.
        """
        if not self.ruta.exists():
            print(f"[aviso] No existe la traza {self.ruta.name}.")
            return

        async with aiofiles.open(self.ruta, encoding="utf-8") as fh:
            crudo = await fh.read()

        eventos: list[dict[str, Any]] = []
        for linea in crudo.splitlines():
            if not linea.strip():
                continue
            try:
                evento = json.loads(linea)
            except json.JSONDecodeError:
                continue
            if isinstance(evento, dict):
                eventos.append(evento)

        if not eventos:
            print("[aviso] La traza está vacía.")
            return

        print(_separador("="))
        print(f" Traza: {self.ruta.name}")
        cabecera = eventos[0]
        if cabecera.get("tipo") == "sesion":
            parametros = cabecera.get("parametros") or {}
            print(f" Sesión: {cabecera.get('thread_id')} · {cabecera.get('ts')}")
            print(
                f" Config: {parametros.get('model_name')} · iteraciones="
                f"{parametros.get('max_iterations_agentes')} · intentos máx="
                f"{parametros.get('max_intentos_correccion')}"
            )
        print(_separador("="))

        turno_actual: Any = None
        for evento in eventos:
            if evento.get("tipo") == "sesion":
                continue
            turno = evento.get("turno")
            if turno is not None and turno != turno_actual:
                turno_actual = turno
                print(f"\n---------- Turno {turno} ----------")
            for linea in _formatear_evento(evento):
                print(linea)
        print()


_ETIQUETAS = {
    "entrada_usuario": "ENTRADA",
    "pensamiento": "SUPERVISOR",
    "accion": "ACCIÓN",
    "observacion": "OBSERVACIÓN",
    "contribucion": "CONTRIBUCIÓN",
    "respuesta_final": "RESPUESTA",
    "error": "ERROR",
}

ANCHO_LINEA = 96
INDENTACION = 14
LIMITE_RESULTADO = 400
ANCHO_MAXIMO = 100


def _separador(caracter: str) -> str:
    """Línea decorativa de ancho fijo."""
    return caracter * 55


def _envolver(texto: str, ancho: int = ANCHO_LINEA) -> list[str]:
    """Parte `texto` en líneas legibles; lista vacía si está en blanco."""
    if not texto.strip():
        return []
    lineas: list[str] = []
    for parrafo in texto.splitlines():
        lineas.extend(textwrap.wrap(parrafo, width=ancho) or [""])
    return lineas


def _texto_respuesta(evento: dict[str, Any]) -> str:
    """Texto plano de un evento `respuesta_final` (incluye bloques del AIMessage)."""
    texto = evento.get("texto")
    if isinstance(texto, str) and texto.strip():
        return texto
    contenido = evento.get("contenido")
    if isinstance(contenido, str):
        return contenido
    if isinstance(contenido, list):
        partes = [
            bloque.get("text", "")
            for bloque in contenido
            if isinstance(bloque, dict) and bloque.get("type") == "text"
        ]
        return "\n".join(parte for parte in partes if parte)
    return ""


def _formatear_evento(evento: dict[str, Any]) -> list[str]:
    """Convierte un evento del `.jsonl` en líneas de texto legibles."""
    tipo = str(evento.get("tipo", "?"))
    etiqueta = _ETIQUETAS.get(tipo, tipo.upper())
    marca = f"{str(evento.get('ts', ''))[-8:]}  {etiqueta:<14}"
    resumen = ""
    cuerpo: list[str] = []

    if tipo == "entrada_usuario":
        contenido = str(evento.get("contenido") or "")
        if len(marca) + len(contenido) <= ANCHO_MAXIMO:
            resumen = contenido
        else:
            cuerpo = _envolver(contenido)
    elif tipo == "pensamiento":
        resumen = f"-> {evento.get('ruta')}"
        cuerpo = _envolver(str(evento.get("decision") or ""))
    elif tipo == "accion":
        resumen = f"{evento.get('nodo')} · {evento.get('herramienta')}"
        argumentos = json.dumps(
            evento.get("argumentos") or {}, ensure_ascii=False, default=str
        )
        cuerpo = _envolver(argumentos)
    elif tipo == "observacion":
        resultado = str(evento.get("resultado") or "")
        recorte = resultado[:LIMITE_RESULTADO]
        if len(resultado) > LIMITE_RESULTADO:
            recorte += "[...]"
        resumen = f"{evento.get('herramienta')} · {len(resultado)} caracteres"
        cuerpo = _envolver(recorte)
    elif tipo == "contribucion":
        resumen = f"{evento.get('nodo')} [{evento.get('clase')}]"
        cuerpo = _envolver(str(evento.get("contenido") or ""))
    elif tipo == "respuesta_final":
        cuerpo = _envolver(_texto_respuesta(evento))
    elif tipo == "error":
        cuerpo = _envolver(str(evento.get("mensaje") or ""))
    else:
        cuerpo = _envolver(json.dumps(evento, ensure_ascii=False, default=str))

    primera = f"{marca}{resumen}".rstrip()
    if resumen and len(primera) > ANCHO_MAXIMO:
        primera = marca.rstrip()
        cuerpo = _envolver(resumen) + cuerpo

    lineas = [primera]
    lineas.extend((" " * INDENTACION + linea) if linea else "" for linea in cuerpo)
    return lineas
