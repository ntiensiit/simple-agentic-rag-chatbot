from rag import Chatbot


def run():
    print("Agentic RAG chatbot. Type /exit to quit.")
    bot = Chatbot()
    while True:
        question = input("\nYou: ").strip()
        if question.lower() in {"/exit", "/quit"}:
            return
        if question:
            print("Bot:", bot.ask(question))


if __name__ == "__main__":
    run()
