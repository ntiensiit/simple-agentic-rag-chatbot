from typing import TypedDict
from urllib.parse import quote
from urllib.request import urlopen
import json

from langchain_core.tools import tool
from langchain_ollama import OllamaLLM
from langgraph.graph import END, StateGraph

from rag.config import CHAT_MODEL, OLLAMA_URL, TOP_K


class State(TypedDict, total=False):
    question: str
    history: str
    query: str
    context: str
    answer: str
    route: str
    retry: int


class Agent:
    def __init__(self, engine):
        self.engine = engine
        self.llm = OllamaLLM(model=CHAT_MODEL, base_url=OLLAMA_URL, temperature=0)
        self.history = []
        self.graph = self._build_graph()

    def answer(self, question: str):
        state = {"question": question, "history": self._history(), "retry": 0}
        result = self.graph.invoke(state)
        self.history.extend(["User: " + question, "Assistant: " + result["answer"]])
        return result["answer"]

    def _build_graph(self):
        graph = StateGraph(State)
        for name, node in [("plan", self._plan), ("rewrite", self._rewrite), ("retrieve", self._retrieve), ("grade_docs", self._grade_docs), ("web", self._web), ("answer", self._answer), ("grade_answer", self._grade_answer), ("check_hallucination", self._check_hallucination)]:
            graph.add_node(name, node)
        graph.set_entry_point("plan")
        graph.add_conditional_edges("plan", lambda s: s["route"], {"local": "rewrite", "web": "web"})
        graph.add_edge("rewrite", "retrieve")
        graph.add_edge("retrieve", "grade_docs")
        graph.add_conditional_edges("grade_docs", self._doc_route, {"answer": "answer", "retry": "rewrite", "web": "web"})
        graph.add_edge("web", "answer")
        graph.add_edge("answer", "grade_answer")
        graph.add_conditional_edges("grade_answer", self._answer_route, {"check": "check_hallucination", "retry": "rewrite", "done": END})
        graph.add_conditional_edges("check_hallucination", self._hallucination_route, {"done": END, "retry": "rewrite"})
        return graph.compile()

    def _plan(self, state):
        prompt = "Planner: choose local for document questions or web for current external questions. Reply local or web.\n" + state["question"]
        return {"route": "web" if "web" in self.llm.invoke(prompt).lower() else "local"}

    def _rewrite(self, state):
        if not state["history"]:
            return {"query": state["question"]}
        prompt = "Researcher: rewrite as a standalone search query. Return only the query.\n" + state["history"] + "\nCurrent: " + state["question"]
        return {"query": self.llm.invoke(prompt).strip()}

    def _retrieve(self, state):
        return {"context": retrieve_documents.invoke({"query": state["query"], "engine": self.engine})}

    def _grade_docs(self, state):
        prompt = "Document grader: reply yes if context supports the question, otherwise no.\n" + state.get("context", "") + "\nQuestion: " + state["question"]
        return {"route": "answer" if "yes" in self.llm.invoke(prompt).lower() else "retry"}

    def _doc_route(self, state):
        if state["route"] == "retry" and state["retry"] < 1:
            state["retry"] += 1
            return "retry"
        return "answer" if state.get("context") else "web"

    def _web(self, state):
        try:
            data = urlopen("https://api.duckduckgo.com/?q=" + quote(state["question"]) + "&format=json", timeout=8).read()
            result = json.loads(data).get("AbstractText", "")
        except Exception:
            result = "No web result is available."
        return {"context": result}

    def _answer(self, state):
        prompt = "Verifier: answer briefly using only context. If unsupported, say you do not know.\nHistory:\n" + state["history"] + "\nContext:\n" + state.get("context", "") + "\nQuestion: " + state["question"]
        return {"answer": self.llm.invoke(prompt)}

    def _grade_answer(self, state):
        prompt = "Answer grader: reply yes if answer addresses the question using context, otherwise no.\n" + state["answer"] + "\n" + state.get("context", "")
        return {"route": "check" if "yes" in self.llm.invoke(prompt).lower() else "retry"}

    def _answer_route(self, state):
        return "check" if state["route"] == "check" else self._retry_route(state)

    def _hallucination_route(self, state):
        return "done" if state["route"] == "done" else self._retry_route(state)

    def _retry_route(self, state):
        state["retry"] += 1
        return "retry" if state["retry"] <= 2 else "done"

    def _check_hallucination(self, state):
        prompt = "Hallucination detector: reply yes if every factual claim is supported by context, otherwise no.\n" + state["answer"] + "\n" + state.get("context", "")
        return {"route": "done" if "yes" in self.llm.invoke(prompt).lower() else "retry"}

    def _history(self):
        return "\n".join(self.history[-6:])


@tool
def retrieve_documents(query: str, engine):
    """Search the local FAISS index."""
    hits = engine.search(query, TOP_K)
    return "\n\n".join(doc.page_content for doc, score in hits if score < 500)
