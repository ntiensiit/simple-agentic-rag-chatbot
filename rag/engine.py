from pathlib import Path

from llama_index.core import Settings, StorageContext, VectorStoreIndex, load_index_from_storage
from llama_index.core.node_parser import SentenceSplitter
from llama_index.core import SimpleDirectoryReader
from llama_index.embeddings.ollama import OllamaEmbedding
from llama_index.llms.ollama import Ollama
from llama_index.vector_stores.faiss import FaissVectorStore
import faiss

from rag.config import CHAT_MODEL, EMBED_MODEL, OLLAMA_URL, STORAGE_DIR


class RagEngine:
    def __init__(self, data_dir: Path):
        self._configure()
        self.index = self._load_or_build(data_dir)

    def _configure(self):
        Settings.llm = Ollama(model=CHAT_MODEL, base_url=OLLAMA_URL, request_timeout=120)
        Settings.embed_model = OllamaEmbedding(model_name=EMBED_MODEL, base_url=OLLAMA_URL)
        Settings.node_parser = SentenceSplitter(chunk_size=500, chunk_overlap=50)

    def _load_or_build(self, data_dir: Path):
        if self._storage_is_fresh(data_dir):
            vector = FaissVectorStore.from_persist_dir(str(STORAGE_DIR))
            context = StorageContext.from_defaults(persist_dir=str(STORAGE_DIR), vector_store=vector)
            return load_index_from_storage(context)
        return self._build(data_dir)

    def _storage_is_fresh(self, data_dir: Path):
        index_file = STORAGE_DIR / "docstore.json"
        files = list(data_dir.glob("**/*"))
        sources = [file for file in files if file.is_file()]
        return index_file.exists() and all(file.stat().st_mtime <= index_file.stat().st_mtime for file in sources)

    def _build(self, data_dir: Path):
        documents = SimpleDirectoryReader(str(data_dir)).load_data()
        vector = FaissVectorStore(faiss_index=faiss.IndexFlatL2(768))
        context = StorageContext.from_defaults(vector_store=vector)
        index = VectorStoreIndex.from_documents(documents, storage_context=context)
        index.storage_context.persist(persist_dir=str(STORAGE_DIR))
        return index

    def search(self, query: str, top_k: int):
        return self.index.as_retriever(similarity_top_k=top_k).retrieve(query)
