from __future__ import annotations

from .config import RAGSettings
from .ingestion import DoclingIngestor
from .models import INDEX_SCHEMA_VERSION
from .providers import BGEReranker, OpenAIEmbeddingProvider, PassthroughReranker
from .service import KnowledgeRAG
from .storage import ChromaVectorStore


def create_knowledge_rag(settings: RAGSettings | None = None) -> KnowledgeRAG:
    """Composition root for the RAG side of RAGAgent."""

    settings = settings or RAGSettings.from_env()
    ingestor = DoclingIngestor(
        embedding_model=settings.embedding_model,
        chunk_max_tokens=settings.chunk_max_tokens,
        pdf_ocr_mode=settings.pdf_ocr_mode,
        strict_conversion=settings.strict_conversion,
    )
    embeddings = OpenAIEmbeddingProvider(
        api_key=settings.embedding_api_key,
        base_url=settings.embedding_base_url,
        model=settings.embedding_model,
    )
    if settings.enable_reranker:
        reranker = BGEReranker(
            model_name=settings.reranker_model,
            use_fp16=settings.reranker_use_fp16,
            min_score=settings.reranker_min_score,
        )
    else:
        reranker = PassthroughReranker()
    store = ChromaVectorStore(
        persist_dir=settings.chroma_dir,
        collection_name=settings.collection_name,
        embedding_model=settings.embedding_model,
        index_schema_version=INDEX_SCHEMA_VERSION,
    )
    return KnowledgeRAG(
        settings=settings,
        ingestor=ingestor,
        embeddings=embeddings,
        reranker=reranker,
        store=store,
    )
