

from nodes import retrieve_node,web_search_node,filter_documents_node,fallback_node,rag_node,question_router_node,should_generate,hallucination_and_answer_relevance_check
from langgraph.graph import START , END , StateGraph
from typing import TypedDict
from langchain_core.messages import BaseMessage
from langchain_core.documents import Document

class AgentState(TypedDict):
    """The dictionary keeps track of the data required by the various nodes in the graph"""
    query : str
    generation : str
    chat_history : list[BaseMessage]
    documents : list[Document]


workflow = StateGraph(AgentState)
workflow.add_node("VectorStore", retrieve_node)
workflow.add_node("SearchEngine", web_search_node)
workflow.add_node("filter_docs", filter_documents_node)
workflow.add_node("fallback", fallback_node)
workflow.add_node("rag", rag_node)

workflow.set_conditional_entry_point(
    question_router_node,
    {
        "llm_fallback": "fallback",
        "VectorStore": "VectorStore",
        "SearchEngine": "SearchEngine",
    },
)

workflow.add_edge("VectorStore", "filter_docs")
workflow.add_edge("SearchEngine", "filter_docs")
workflow.add_conditional_edges(
    "filter_docs", should_generate, {"SearchEngine": "SearchEngine", "generate": "rag"}
)
workflow.add_conditional_edges(
    "rag",
    hallucination_and_answer_relevance_check,
    {"useful": END, "not useful": "SearchEngine", "generate": "rag"},
)

workflow.add_edge("fallback", END)

graph = workflow.compile()


