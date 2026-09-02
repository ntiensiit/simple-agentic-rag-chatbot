# Simple Agentic RAG Chatbot

Terminal chatbot that routes questions to either direct LLM answers or document search (RAG).

**Stack:** LangChain + LlamaIndex + FAISS + Ollama (`llama3.2`, `nomic-embed-text`)

## Setup

```bash
python -m venv .venv
.venv\Scripts\activate        # Windows
pip install -r requirements.txt
ollama pull llama3.2
ollama pull nomic-embed-text
```

## Run

```bash
python main.py
```

Add `.txt` or `.md` files to `data/`, then type `/reload`.

## Commands

| Command | Action |
|---------|--------|
| `/reload` | Re-index documents |
| `/clear` | Clear screen |
| `/quit` | Exit |
