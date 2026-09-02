from rag.indexer import build_index, load_index

__all__ = ["build_index", "load_index", "create_agent"]


def __getattr__(name: str):
    if name == "create_agent":
        from rag.agent import create_agent
        return create_agent
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
