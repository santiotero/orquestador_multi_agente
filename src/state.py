"""Estado personalizado del grafo: MessagesState + contribuciones + control."""

from __future__ import annotations

import operator
from typing import Annotated, Any

from langgraph.graph import MessagesState


def texto_plano(contenido: Any) -> str:
    """Extrae el texto visible de un contenido de mensaje (str o lista de bloques)."""
    if isinstance(contenido, str):
        return contenido
    if isinstance(contenido, list):
        partes = [
            bloque["text"]
            for bloque in contenido
            if isinstance(bloque, dict)
            and isinstance(bloque.get("text"), str)
            and bloque.get("type", "text") == "text"
        ]
        return "\n".join(partes)
    if isinstance(contenido, dict):
        texto = contenido.get("text")
        if isinstance(texto, str):
            return texto
    return "" if contenido is None else str(contenido)


class CustomState(MessagesState):
    """Estado global del orquestador.

    Campos y reducers:

    * `messages` (`add_messages`): historial de la sesión. El último mensaje
      humano de cada turno es la pregunta del usuario.
    * `contribuciones` (`operator.add`): aportes de los agentes especialistas,
      cada uno etiquetado con la pregunta que lo generó.
    * `intentos_correccion` (sobrescribe): contador de intentos de corrección.
    * `instruccion_actual` (sobrescribe): instrucción del supervisor para el
      agente que viene.
    * `next_agent` (sobrescribe): campo transitorio con la ruta elegida por el
      supervisor; lo consume la arista condicional.
    """

    contribuciones: Annotated[list[dict[str, Any]], operator.add]
    intentos_correccion: int
    instruccion_actual: str
    next_agent: str


def pregunta_actual(state: Any) -> str:
    """Pregunta del usuario en curso: el último mensaje humano del historial.

    En el primer turno coincide con `state["messages"][0].content`.
    """
    mensajes = list(state.get("messages") or [])
    for mensaje in reversed(mensajes):
        if getattr(mensaje, "type", None) == "human":
            texto = texto_plano(getattr(mensaje, "content", ""))
            if texto.strip():
                return texto
    if mensajes:
        return texto_plano(getattr(mensajes[0], "content", ""))
    return ""


def contribuciones_del_turno(state: Any) -> list[dict[str, Any]]:
    """Contribuciones asociadas a la pregunta en curso.

    Cada agente etiqueta su aporte con la pregunta que lo generó
    (`metadata["pregunta"]`), de modo que los turnos anteriores de la misma
    sesión no contaminan la evaluación del supervisor ni la redacción final.
    """
    pregunta = pregunta_actual(state)
    contribuciones = list(state.get("contribuciones") or [])
    return [
        contribucion
        for contribucion in contribuciones
        if (contribucion.get("metadata") or {}).get("pregunta") == pregunta
    ]
