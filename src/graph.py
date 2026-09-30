"""StateGraph del orquestador: nodos, aristas condicionales y compilación."""

from __future__ import annotations

from langgraph.checkpoint.base import BaseCheckpointSaver
from langgraph.graph import END, START, StateGraph
from langgraph.graph.state import CompiledStateGraph
from langgraph.prebuilt import ToolNode

from src.agents.analista import crear_agente_analista
from src.agents.investigador import crear_agente_investigador
from src.nodes import crear_nodo_sintesis, crear_nodo_supervisor
from src.providers.gemini import GeminiClient
from src.schema.models import ConfigParams
from src.state import CustomState
from src.tools import crear_tools

_RUTAS: dict[str, str] = {
    "investigador": "investigador",
    "analista": "analista",
    "sintesis": "sintesis",
}


def decidir_ruta(state: CustomState) -> str:
    """Arista condicional: consume `next_agent`, la decisión transitoria del supervisor."""
    return state.get("next_agent") or "sintesis"


def construir_grafo(
    cliente: GeminiClient,
    cfg: ConfigParams,
    saver: BaseCheckpointSaver,
) -> CompiledStateGraph:
    """Compila el grafo:

    START → supervisor ─┬→ investigador → supervisor
                        ├→ analista      → supervisor
                        └→ sintesis      → END

    Solo `sintesis` conecta con `END`.
    """
    tools = crear_tools(cfg)

    builder = StateGraph(CustomState)
    builder.add_node("supervisor", crear_nodo_supervisor(cliente, cfg))
    builder.add_node("investigador", crear_agente_investigador(cliente, cfg))
    builder.add_node("analista", crear_agente_analista(cliente, cfg))
    builder.add_node("sintesis", crear_nodo_sintesis(cliente, cfg))
    builder.add_node("tools", ToolNode(tools, handle_tool_errors=True))

    builder.add_edge(START, "supervisor")
    builder.add_conditional_edges("supervisor", decidir_ruta, _RUTAS)
    builder.add_edge("investigador", "supervisor")
    builder.add_edge("analista", "supervisor")
    builder.add_edge("sintesis", END)

    return builder.compile(checkpointer=saver)
