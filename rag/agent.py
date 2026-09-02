"""Simple agentic RAG: route to search or direct answer."""

from langchain_core.messages import HumanMessage, SystemMessage
from langchain_ollama import ChatOllama

from rag.retriever import DocumentRetriever

LLM_MODEL = "llama3.2"
OLLAMA_BASE_URL = "http://127.0.0.1:11434"

ROUTER_PROMPT = """You route messages for a document-backed chatbot.
Reply with exactly one word: SEARCH or DIRECT.

SEARCH = questions about this project, its docs, setup, commands, architecture, or uploaded content
DIRECT = greetings, small talk, or unrelated general knowledge"""

ANSWER_PROMPT = "You are a helpful assistant. Answer concisely."

RAG_PROMPT = "Answer using the provided context. Be concise. If the context is insufficient, say so."


class ChatAgent:
    def __init__(self, retriever: DocumentRetriever):
        self.retriever = retriever
        self.llm = ChatOllama(model=LLM_MODEL, temperature=0.2, base_url=OLLAMA_BASE_URL)

    def invoke(self, inputs: dict) -> dict:
        query = inputs["input"]
        route = self.llm.invoke(
            [SystemMessage(content=ROUTER_PROMPT), HumanMessage(content=query)]
        ).content.strip().upper()

        if "SEARCH" in route:
            context = self.retriever.search(query)
            answer = self.llm.invoke(
                [
                    SystemMessage(content=RAG_PROMPT),
                    HumanMessage(content=f"Context:\n{context}\n\nQuestion: {query}"),
                ]
            ).content
        else:
            answer = self.llm.invoke(
                [SystemMessage(content=ANSWER_PROMPT), HumanMessage(content=query)]
            ).content

        return {"output": answer}


def create_agent(retriever: DocumentRetriever) -> ChatAgent:
    return ChatAgent(retriever)
