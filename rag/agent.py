from langchain_ollama import OllamaLLM
from langgraph.graph import END, StateGraph
from typing import TypedDict

from rag.config import CHAT_MODEL, OLLAMA_URL, TOP_K


class State(TypedDict, total=False):
    question: str
    query: str
    history: str
    context: str
    answer: str


class Agent:
    def __init__(self, engine):
        self.engine = engine
        self.llm = OllamaLLM(model=CHAT_MODEL, base_url=OLLAMA_URL, temperature=0)
        self.history = []
        self.graph = self._build_graph()

    def answer(self, question: str):
        state = {"question": question, "history": self._history_text()}
        result = self.graph.invoke(state)
        self._remember(question, result["answer"])
        return result["answer"]

    def _build_graph(self):
        graph = StateGraph(State)
        graph.add_node("rewrite", self._rewrite)
        graph.add_node("retrieve", self._retrieve)
        graph.add_node("answer", self._answer)
        graph.set_entry_point("rewrite")
        graph.add_edge("rewrite", "retrieve")
        graph.add_edge("retrieve", "answer")
        graph.add_edge("answer", END)
        return graph.compile()

    def _rewrite(self, state):
        if not state["history"]:
            return {"query": state["question"]}
        prompt = "Rewrite as a standalone search query. Return only the query.\n" + state["history"] + "\nCurrent: " + state["question"]
        return {"query": self.llm.invoke(prompt).strip()}

    def _retrieve(self, state):
        hits = self.engine.search(state["query"], TOP_K)
        context = "\n\n".join(doc.page_content for doc, score in hits if score < 500)
        return {"context": context}

    def _answer(self, state):
        context = state.get("context", "")
        prompt = "Answer using context and conversation. If missing, say you do not know.\nContext:\n" + context + "\nConversation:\n" + state["history"] + "\nUser: " + state["question"]
        return {"answer": self.llm.invoke(prompt)}

    def _history_text(self):
        return "\n".join(self.history[-6:])

    def _remember(self, question: str, answer: str):
        self.history.extend(["User: " + question, "Assistant: " + answer])
