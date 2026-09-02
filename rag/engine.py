from pathlib import Path
import hashlib
import json

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
        manifest_file = STORAGE_DIR / "manifest.json"
        if not index_file.exists() or not manifest_file.exists():
            return False
        return self._manifest(data_dir) == json.loads(manifest_file.read_text())

    def _build(self, data_dir: Path):
        documents = [TextLoader(str(file), encoding="utf-8").load()[0] for file in data_dir.glob("**/*.txt")]
        chunks = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=50).split_documents(documents)
        index = FAISS.from_documents(chunks, self.embeddings)
        STORAGE_DIR.mkdir(exist_ok=True)
        index.save_local(str(STORAGE_DIR))
        manifest = self._manifest(data_dir)
        (STORAGE_DIR / "manifest.json").write_text(json.dumps(manifest, indent=2))
        return index

    def _manifest(self, data_dir: Path):
        files = sorted(file for file in data_dir.glob("**/*.txt") if file.is_file())
        return [self._file_record(file, data_dir) for file in files]

    def _file_record(self, file: Path, data_dir: Path):
        stat = file.stat()
        return {"path": str(file.relative_to(data_dir)), "size": stat.st_size, "mtime": stat.st_mtime, "hash": self._file_hash(file)}

    def _file_hash(self, file: Path):
        digest = hashlib.sha256()
        with file.open("rb") as source:
            for block in iter(lambda: source.read(65536), b""):
                digest.update(block)
        return digest.hexdigest()

    def search(self, query: str, top_k: int):
        return self.index.similarity_search_with_score(query, k=top_k)
