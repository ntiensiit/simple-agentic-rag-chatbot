from pathlib import Path

from langchain_community.document_loaders import TextLoader
from langchain_ollama import OllamaEmbeddings
from langchain_community.vectorstores import FAISS
from langchain_text_splitters import RecursiveCharacterTextSplitter

from rag.config import EMBED_MODEL, OLLAMA_URL, STORAGE_DIR


class RagEngine:
    def __init__(self, data_dir: Path):
        self.embeddings = OllamaEmbeddings(model=EMBED_MODEL, base_url=OLLAMA_URL)
        self.index = self._load_or_build(data_dir)

    def _load_or_build(self, data_dir: Path):
        if self._storage_is_fresh(data_dir):
            return FAISS.load_local(str(STORAGE_DIR), self.embeddings, allow_dangerous_deserialization=True)
        return self._build(data_dir)

    def _storage_is_fresh(self, data_dir: Path):
        index_file = STORAGE_DIR / "index.faiss"
        files = list(data_dir.glob("**/*"))
        sources = [file for file in files if file.is_file()]
        return index_file.exists() and all(file.stat().st_mtime <= index_file.stat().st_mtime for file in sources)

    def _build(self, data_dir: Path):
        documents = [TextLoader(str(file), encoding="utf-8").load()[0] for file in data_dir.glob("**/*.txt")]
        chunks = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=50).split_documents(documents)
        index = FAISS.from_documents(chunks, self.embeddings)
        STORAGE_DIR.mkdir(exist_ok=True)
        index.save_local(str(STORAGE_DIR))
        return index

    def search(self, query: str, top_k: int):
        return self.index.similarity_search_with_score(query, k=top_k)
