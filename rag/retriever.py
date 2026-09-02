"""Document retrieval via LlamaIndex vector retriever."""

from llama_index.core import VectorStoreIndex
from llama_index.core.retrievers import VectorIndexRetriever


class DocumentRetriever:
    """Thin wrapper around a LlamaIndex retriever for LangChain tool use."""

    def __init__(self, index: VectorStoreIndex, top_k: int = 4):
        self._retriever = VectorIndexRetriever(index=index, similarity_top_k=top_k)

    def search(self, query: str) -> str:
        nodes = self._retriever.retrieve(query)
        if not nodes:
            return "No relevant documents found."

        parts = []
        for i, node in enumerate(nodes, 1):
            score = f" (score: {node.score:.3f})" if node.score is not None else ""
            parts.append(f"[{i}]{score}\n{node.get_content()}")
        return "\n\n".join(parts)
