"""Structure-aware RAG components integrated with the local Agent project."""

from .config import RAGSettings
from .factory import create_knowledge_rag
from .models import DocumentChunk, IndexReport, SearchHit
from .service import KnowledgeRAG

__all__ = [
    "DocumentChunk",
    "IndexReport",
    "KnowledgeRAG",
    "RAGSettings",
    "SearchHit",
    "create_knowledge_rag",
]
