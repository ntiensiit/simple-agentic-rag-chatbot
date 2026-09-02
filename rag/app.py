from rag.agent import Agent
from rag.config import DATA_DIR
from rag.engine import RagEngine


class Chatbot:
    def __init__(self):
        self.agent = Agent(RagEngine(DATA_DIR))

    def ask(self, question: str):
        return self.agent.answer(question)
