from __future__ import annotations

import hashlib
import json
import os
from dataclasses import dataclass
from pathlib import Path

from dotenv import load_dotenv

from .models import INDEX_SCHEMA_VERSION


def _env_int(name: str, default: int) -> int:
    value = os.getenv(name)
    return int(value) if value else default


def _env_bool(name: str, default: bool) -> bool:
    value = os.getenv(name)
    if value is None:
        return default
    return value.strip().lower() in {"1", "true", "yes", "on"}


def _optional_env(name: str) -> str | None:
    value = os.getenv(name)
    return value.strip() if value and value.strip() else None


def _env_optional_float(name: str, default: float | None) -> float | None:
    value = os.getenv(name)
    if value is None or not value.strip():
        return default
    if value.strip().lower() in {"none", "off", "disabled"}:
        return None
    return float(value)


@dataclass(frozen=True, slots=True)
class RAGSettings:
    project_root: Path
    chroma_dir: Path
    collection_name: str

    embedding_api_key: str
    embedding_base_url: str | None
    embedding_model: str

    chunk_max_tokens: int
    embedding_batch_size: int
    dense_k: int
    keyword_k: int
    fusion_k: int
    final_k: int
    rrf_constant: int
    jieba_user_dict: Path | None

    enable_reranker: bool
    reranker_model: str
    reranker_use_fp16: bool
    reranker_min_score: float | None

    pdf_ocr_mode: str
    strict_conversion: bool
    prune_missing_sources: bool

    @classmethod
    def from_env(cls) -> RAGSettings:
        project_root = Path(__file__).resolve().parents[2]
        load_dotenv(project_root / ".env")

        common_key = _optional_env("OPENAI_API_KEY") or ""
        embedding_key = (
            _optional_env("EMBEDDING_API_KEY")
            or _optional_env("EMBEDDING_KEY")
            or common_key
        )
        settings = cls(
            project_root=project_root,
            chroma_dir=Path(
                os.getenv(
                    "RAG_CHROMA_DIR",
                    str(project_root / "rag_chroma"),
                )
            ).expanduser().resolve(),
            collection_name=os.getenv(
                "RAG_CHROMA_COLLECTION",
                "agent_knowledge_v2",
            ),
            embedding_api_key=embedding_key,
            embedding_base_url=(
                _optional_env("EMBEDDING_BASE_URL")
                or _optional_env("OPENAI_BASE_URL")
            ),
            embedding_model=os.getenv(
                "EMBEDDING_MODEL",
                "BAAI/bge-large-zh-v1.5",
            ),
            chunk_max_tokens=_env_int("RAG_CHUNK_MAX_TOKENS", 700),
            embedding_batch_size=_env_int("RAG_EMBEDDING_BATCH_SIZE", 64),
            dense_k=_env_int("RAG_DENSE_K", 30),
            keyword_k=_env_int("RAG_KEYWORD_K", 30),
            fusion_k=_env_int("RAG_FUSION_K", 30),
            final_k=_env_int("RAG_FINAL_K", 6),
            rrf_constant=_env_int("RAG_RRF_CONSTANT", 60),
            jieba_user_dict=(
                Path(value).expanduser().resolve()
                if (value := _optional_env("JIEBA_USER_DICT"))
                else None
            ),
            enable_reranker=_env_bool("RAG_ENABLE_RERANKER", True),
            reranker_model=os.getenv(
                "RAG_RERANKER_MODEL",
                "BAAI/bge-reranker-v2-m3",
            ),
            reranker_use_fp16=_env_bool("RAG_RERANKER_USE_FP16", False),
            reranker_min_score=_env_optional_float(
                "RAG_RERANKER_MIN_SCORE",
                0.1,
            ),
            pdf_ocr_mode=os.getenv("RAG_PDF_OCR_MODE", "default").lower(),
            strict_conversion=_env_bool("RAG_STRICT_CONVERSION", True),
            prune_missing_sources=_env_bool("RAG_PRUNE_MISSING_SOURCES", True),
        )
        settings.validate()
        return settings

    def index_fingerprint(self, content_hash: str) -> str:
        profile = {
            "schema": INDEX_SCHEMA_VERSION,
            "content_hash": content_hash,
            "embedding_model": self.embedding_model,
            "chunk_max_tokens": self.chunk_max_tokens,
            "pdf_ocr_mode": self.pdf_ocr_mode,
            "strict_conversion": self.strict_conversion,
        }
        payload = json.dumps(profile, sort_keys=True, separators=(",", ":"))
        return hashlib.sha256(payload.encode()).hexdigest()

    def validate(self) -> None:
        if not self.embedding_api_key:
            raise ValueError(
                "缺少 EMBEDDING_API_KEY、EMBEDDING_KEY 或 OPENAI_API_KEY"
            )
        if self.pdf_ocr_mode not in {"default", "full_page"}:
            raise ValueError("RAG_PDF_OCR_MODE 只能是 default 或 full_page")

        positive_values = {
            "RAG_CHUNK_MAX_TOKENS": self.chunk_max_tokens,
            "RAG_EMBEDDING_BATCH_SIZE": self.embedding_batch_size,
            "RAG_DENSE_K": self.dense_k,
            "RAG_KEYWORD_K": self.keyword_k,
            "RAG_FUSION_K": self.fusion_k,
            "RAG_FINAL_K": self.final_k,
            "RAG_RRF_CONSTANT": self.rrf_constant,
        }
        for name, value in positive_values.items():
            if value <= 0:
                raise ValueError(f"{name} 必须大于 0")
        if self.final_k > self.fusion_k:
            raise ValueError("RAG_FINAL_K 不能大于 RAG_FUSION_K")
        if (
            self.reranker_min_score is not None
            and not 0 <= self.reranker_min_score <= 1
        ):
            raise ValueError("RAG_RERANKER_MIN_SCORE 必须在 0 到 1 之间")
