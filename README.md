# Simple agentic RAG chatbot

## Requirements

- Python 3.10 or newer
- Ollama
- Ollama models: `llama3.2` and `nomic-embed-text`

## Setup

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install -r requirements.txt
ollama pull llama3.2
ollama pull nomic-embed-text
```

Add `.txt` files to `data`, then run:

```powershell
python main.py
```

The first run builds `storage`. Delete `storage` after changing documents.
