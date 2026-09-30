"""Nodos puros del grafo: supervisor (pasamanos) y síntesis (redacción final)."""

from __future__ import annotations

import json
from typing import Any

from langchain_core.messages import AIMessage, HumanMessage
from langchain_core.runnables import RunnableConfig

from src.providers.gemini import GeminiClient, invocar_con_reintentos
from src.schema.models import ConfigParams, SupervisorOutput
from src.state import CustomState, contribuciones_del_turno, pregunta_actual

PROMPT_SUPERVISOR = """\
Eres el supervisor de un sistema multi-agente. Analiza las contribuciones acumuladas y decide el siguiente paso.

CONTEXTO:
- Pregunta original: {pregunta_original}
- Contribuciones: {contribuciones}
- Intentos de corrección: {intentos_correccion}
- Máximo permitido: {max_intentos_correccion}

REGLAS:
1. Si el analista cometió un error de cálculo o el investigador trajo datos incompletos, envía una instrucción de ajuste.
2. Si intentos_correccion >= max_intentos_correccion, DEBES elegir next='sintesis'.
3. Si la tarea está resuelta, elige next='sintesis'.
4. Si necesitas más información, elige next='investigador'.
5. Si necesitas verificar cálculos, elige next='analista'.
6. Si todavía no hay contribuciones, empieza por next='investigador' con una instrucción concreta.
7. Si la pregunta contiene números, porcentajes, precios, descuentos u operaciones matemáticas (inclusive signos % o cálculos implícitos), DEBES elegir next='analista' antes que 'sintesis', salvo que en este turno ya exista una contribución del analista. Una mención a un resultado numérico dentro de la contribución del investigador NO cuenta como verificación: el analista debe confirmarlo con su herramienta. La instrucción para el analista debe llevar los números exactos a calcular.
8. Una contribución que empiece con "ERROR" o que diga "límite de iteraciones" está incompleta: repárala con otra instrucción o cierra con 'sintesis' si ya se agotaron los intentos.

OUTPUT (Pydantic):
- next: Literal['investigador', 'analista', 'sintesis']
- instruccion: str (instrucción ultra-específica para el siguiente agente)
- razon: str (justificación de tu decisión)
"""

PROMPT_SINTESIS = """\
Eres el redactor final de un sistema multi-agente. Tu tarea es generar una respuesta unificada y coherente.

PREGUNTA ORIGINAL DEL USUARIO:
{pregunta_original}

CONTRIBUCIONES DE LOS AGENTES:
{contribuciones}

REGLAS:
1. Sintetiza la información de forma clara y directa.
2. Si faltan datos o hay inconsistencias, indícalo.
3. Responde en español.
4. No agregues información que no esté en las contribuciones.
5. Formatea la respuesta para lectura fácil (listas, tablas si aplica).
"""

INSTRUCCION_FINAL = "Generar respuesta final con la información disponible."


def _formato_contribuciones(contribuciones: list[dict[str, Any]]) -> str:
    if not contribuciones:
        return "(todavía no hay contribuciones)"
    return json.dumps(contribuciones, ensure_ascii=False, indent=2, default=str)


def crear_nodo_supervisor(cliente: GeminiClient, cfg: ConfigParams) -> Any:
    """Nodo `supervisor`: función pura async que delega toda la evaluación al LLM.

    El LLM lee las contribuciones, detecta errores y devuelve `SupervisorOutput`
    (`next`, `instruccion`, `razon`). El código solo:

    * registra los intentos de corrección (`intentos_correccion`),
    * aplica la red de seguridad que evita bucles infinitos si el LLM falla o
      alucina (llegó al límite → `sintesis`).

    Mejora 1: la respuesta JSON del LLM NO se agrega a `state["messages"]`, para
    no ensuciar el contexto de los agentes especialistas.
    Mejora 2: `next_agent` es un campo transitorio (sobrescribe, no acumula); lo
    consume la arista condicional.
    """
    supervisor_llm = cliente.estructurado(SupervisorOutput)

    async def nodo_supervisor(state: CustomState, config: RunnableConfig) -> dict[str, Any]:
        pregunta = pregunta_actual(state)
        contribuciones = contribuciones_del_turno(state)
        intentos = int(state.get("intentos_correccion") or 0)

        prompt = PROMPT_SUPERVISOR.format(
            pregunta_original=pregunta,
            contribuciones=_formato_contribuciones(contribuciones),
            intentos_correccion=intentos,
            max_intentos_correccion=cfg.max_intentos_correccion,
        )

        try:
            resultado = await invocar_con_reintentos(
                supervisor_llm,
                [HumanMessage(prompt)],
                intentos=2,
            )
        except Exception as exc:
            raise RuntimeError(
                f"El supervisor no pudo invocar al LLM tras reintentar: {type(exc).__name__}: {exc}"
            ) from exc

        nuevo_valor_intentos = intentos
        if resultado.next in ("investigador", "analista"):
            nuevo_valor_intentos += 1

        if nuevo_valor_intentos >= cfg.max_intentos_correccion:
            return {
                "intentos_correccion": nuevo_valor_intentos,
                "next_agent": "sintesis",
                "instruccion_actual": INSTRUCCION_FINAL,
            }

        return {
            "intentos_correccion": nuevo_valor_intentos,
            "next_agent": resultado.next,
            "instruccion_actual": resultado.instruccion,
        }

    return nodo_supervisor


def crear_nodo_sintesis(cliente: GeminiClient, cfg: ConfigParams) -> Any:
    """Nodo `sintesis`: función pura async que redacta la respuesta unificada.

    Único nodo conectado con `END`. No usa herramientas ni bucles internos:
    lee la pregunta y las contribuciones del turno y devuelve el texto final
    como mensaje del asistente.
    """
    llm = cliente.modelo

    async def nodo_sintesis(state: CustomState, config: RunnableConfig) -> dict[str, Any]:
        pregunta = pregunta_actual(state)
        contribuciones = contribuciones_del_turno(state)

        prompt = PROMPT_SINTESIS.format(
            pregunta_original=pregunta,
            contribuciones=_formato_contribuciones(contribuciones),
        )

        try:
            respuesta = await invocar_con_reintentos(
                llm,
                [HumanMessage(prompt)],
                intentos=2,
            )
        except Exception as exc:
            raise RuntimeError(
                f"La síntesis no pudo invocar al LLM tras reintentar: {type(exc).__name__}: {exc}"
            ) from exc

        contenido = respuesta.content if isinstance(respuesta, AIMessage) else respuesta
        return {"messages": [AIMessage(content=contenido, name="sintesis")]}

    return nodo_sintesis
