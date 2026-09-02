from langchain_ollama import OllamaLLM

from rag.config import CHAT_MODEL, OLLAMA_URL, TOP_K


class Agent:
    def __init__(self, engine):
        self.engine = engine
        self.llm = OllamaLLM(model=CHAT_MODEL, base_url=OLLAMA_URL, temperature=0)
        self.history = []

    def answer(self, question: str):
        query = self._history_text() + "\nCurrent question: " + question
        hits = self.engine.search(query, TOP_K)
        if hits and hits[0][1] < 500:
            answer = self._grounded(query, hits)
        else:
            answer = self.llm.invoke("Answer briefly and accurately.\n" + query)
        self._remember(question, answer)
        return answer

    def _grounded(self, question: str, hits):
        context = "\n\n".join(hit.page_content for hit, _ in hits)
        prompt = "Answer using context and conversation. If missing, say you do not know.\nContext:\n" + context + "\n" + question
        return self.llm.invoke(prompt)

    def _history_text(self):
        return "\n".join(self.history[-6:])

    def _remember(self, question: str, answer: str):
        self.history.extend(["User: " + question, "Assistant: " + answer])
