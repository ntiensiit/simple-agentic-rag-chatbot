#!/usr/bin/env python3
"""Agentic RAG chatbot TUI — LangChain agent + LlamaIndex/FAISS retrieval."""

import logging
import os
import sys

if sys.platform == "win32":
    for stream in (sys.stdout, sys.stderr):
        if hasattr(stream, "reconfigure"):
            stream.reconfigure(encoding="utf-8")

for name in ("httpx", "httpcore", "llama_index", "faiss"):
    logging.getLogger(name).setLevel(logging.WARNING)

from rag.agent import create_agent
from rag.indexer import DEFAULT_DATA_DIR, build_index, index_exists, load_index
from rag.retriever import DocumentRetriever


def ensure_index() -> None:
    if index_exists():
        return
    if DEFAULT_DATA_DIR.exists() and any(DEFAULT_DATA_DIR.iterdir()):
        build_index()


def reload_index() -> bool:
    if not DEFAULT_DATA_DIR.exists() or not any(DEFAULT_DATA_DIR.iterdir()):
        return False
    build_index()
    return True


def get_retriever() -> DocumentRetriever | None:
    if not index_exists():
        return None
    return DocumentRetriever(load_index())


def run_chat() -> None:
    ensure_index()
    retriever = get_retriever()
    agent = create_agent(retriever) if retriever else None

    while True:
        try:
            user_input = input("You> ").strip()
        except (EOFError, KeyboardInterrupt):
            break

        if not user_input:
            continue

        cmd = user_input.lower()
        if cmd in ("/quit", "/exit", "/q"):
            break
        if cmd == "/clear":
            os.system("cls" if sys.platform == "win32" else "clear")
            continue
        if cmd == "/reload":
            if reload_index():
                retriever = get_retriever()
                agent = create_agent(retriever) if retriever else None
            continue

        if agent is None:
            print("No knowledge base. Add files to data/ and /reload.")
            continue

        try:
            result = agent.invoke({"input": user_input})
            print(f"Assistant> {result.get('output', result)}")
        except Exception as exc:
            print(f"Error: {exc}")


def main() -> None:
    try:
        run_chat()
    except Exception as exc:
        print(f"Error: {exc}")
        sys.exit(1)


if __name__ == "__main__":
    main()
