from langchain_ollama import OllamaLLM

from rag.config import CHAT_MODEL, OLLAMA_URL, TOP_K


class Agent:
    def __init__(self, engine):
        self.engine = engine
        self.llm = OllamaLLM(model=CHAT_MODEL, base_url=OLLAMA_URL, temperature=0)

    def answer(self, question: str):
        hits = self.engine.search(question, TOP_K)
        if hits and hits[0].score < 1.3:
            return self._grounded(question, hits)
        return self.llm.invoke(f"Answer briefly and accurately. Question: {question}")

    def _grounded(self, question: str, hits):
        context = "\n\n".join(hit.node.get_content() for hit in hits)
        prompt = "Answer using only the context. If missing, say you do not know.\nContext:\n" + context + "\nQuestion: " + question
        return self.llm.invoke(prompt)
