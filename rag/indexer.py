"""LlamaIndex + FAISS document indexing with Ollama embeddings."""

from pathlib import Path

import faiss
from llama_index.core import (
    Settings,
    SimpleDirectoryReader,
    StorageContext,
    VectorStoreIndex,
    load_index_from_storage,
)
from llama_index.embeddings.ollama import OllamaEmbedding
from llama_index.vector_stores.faiss import FaissVectorStore

EMBED_MODEL = "nomic-embed-text"
EMBED_DIM = 768
OLLAMA_BASE_URL = "http://127.0.0.1:11434"
DEFAULT_DATA_DIR = Path("data")
DEFAULT_STORAGE_DIR = Path("storage")


def _configure_embed_model() -> OllamaEmbedding:
    embed_model = OllamaEmbedding(model_name=EMBED_MODEL, base_url=OLLAMA_BASE_URL)
    Settings.embed_model = embed_model
    return embed_model


def _create_vector_store() -> FaissVectorStore:
    faiss_index = faiss.IndexFlatL2(EMBED_DIM)
    return FaissVectorStore(faiss_index=faiss_index)


def build_index(
    data_dir: Path = DEFAULT_DATA_DIR,
    storage_dir: Path = DEFAULT_STORAGE_DIR,
) -> VectorStoreIndex:
    """Load documents from *data_dir*, embed them, and persist a FAISS index."""
    _configure_embed_model()
    storage_dir.mkdir(parents=True, exist_ok=True)

    documents = SimpleDirectoryReader(str(data_dir)).load_data()
    if not documents:
        raise ValueError(f"No documents found in {data_dir.resolve()}")

    vector_store = _create_vector_store()
    storage_context = StorageContext.from_defaults(vector_store=vector_store)
    index = VectorStoreIndex.from_documents(documents, storage_context=storage_context)
    index.storage_context.persist(persist_dir=str(storage_dir))
    return index


def load_index(storage_dir: Path = DEFAULT_STORAGE_DIR) -> VectorStoreIndex:
    """Load a previously persisted FAISS index."""
    _configure_embed_model()
    vector_store = FaissVectorStore.from_persist_dir(str(storage_dir))
    storage_context = StorageContext.from_defaults(
        vector_store=vector_store,
        persist_dir=str(storage_dir),
    )
    return load_index_from_storage(storage_context)


def index_exists(storage_dir: Path = DEFAULT_STORAGE_DIR) -> bool:
    return (storage_dir / "docstore.json").exists()
