from __future__ import annotations

import hashlib
from pathlib import Path
from typing import Any

from .models import DocumentChunk
from .provenance import SourceLineMapper, extract_source_locator

SUPPORTED_SUFFIXES = {".pdf", ".docx", ".txt", ".md"}


def discover_files(target: Path) -> list[Path]:
    target = target.expanduser().resolve()
    if target.is_file():
        candidates = [target]
    elif target.is_dir():
        candidates = sorted(path for path in target.rglob("*") if path.is_file())
    else:
        raise FileNotFoundError(f"路径不存在：{target}")

    files = [
        path for path in candidates if path.suffix.lower() in SUPPORTED_SUFFIXES
    ]
    if not files and target.is_file():
        supported = ", ".join(sorted(SUPPORTED_SUFFIXES))
        raise RuntimeError(f"文件格式不受支持，当前支持：{supported}")
    return files


def file_content_hash(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as source:
        while block := source.read(1024 * 1024):
            digest.update(block)
    return digest.hexdigest()


def _chunk_id(
    source: str,
    content_hash: str,
    index_fingerprint: str,
    index: int,
    element_refs: tuple[str, ...],
    text: str,
) -> str:
    value = "\0".join(
        (
            source,
            content_hash,
            index_fingerprint,
            str(index),
            "|".join(element_refs),
            text,
        )
    ).encode()
    return hashlib.sha256(value).hexdigest()[:32]


def _conversion_error_message(path: Path, result: Any) -> str:
    details = []
    for error in result.errors:
        page = f"第 {error.page_no} 页" if error.page_no is not None else "文档级"
        details.append(f"{page}/{error.module_name}: {error.error_message}")
    suffix = "；".join(details) if details else "Docling 未提供具体错误"
    return f"文档解析不完整：{path}；status={result.status.value}；{suffix}"


class DoclingIngestor:
    """Docling parsing with structure-aware, token-aware chunks and provenance."""

    def __init__(
        self,
        embedding_model: str,
        chunk_max_tokens: int,
        pdf_ocr_mode: str = "default",
        strict_conversion: bool = True,
    ) -> None:
        try:
            import tiktoken
            from docling.chunking import HybridChunker
            from docling.datamodel.base_models import InputFormat
            from docling.datamodel.pipeline_options import (
                OcrMode,
                PdfPipelineOptions,
                TableFormerMode,
            )
            from docling.document_converter import DocumentConverter, PdfFormatOption
            from docling_core.transforms.chunker.tokenizer.openai import (
                OpenAITokenizer,
            )
        except ImportError as exc:
            raise RuntimeError(
                "缺少 Docling 或切块依赖：docling、docling-core[chunking-openai]、tiktoken"
            ) from exc

        pdf_options = PdfPipelineOptions()
        pdf_options.do_ocr = True
        pdf_options.do_table_structure = True
        pdf_options.document_timeout = 120
        pdf_options.ocr_options.lang = ["chinese"]
        pdf_options.ocr_options.mode = (
            OcrMode.FULL_PAGE
            if pdf_ocr_mode == "full_page"
            else OcrMode.DEFAULT
        )
        pdf_options.table_structure_options.mode = TableFormerMode.ACCURATE
        pdf_options.table_structure_options.do_cell_matching = True

        self.converter = DocumentConverter(
            format_options={
                InputFormat.PDF: PdfFormatOption(pipeline_options=pdf_options)
            }
        )
        self.strict_conversion = strict_conversion

        try:
            encoding = tiktoken.encoding_for_model(embedding_model)
        except KeyError:
            encoding = tiktoken.get_encoding("cl100k_base")
        tokenizer = OpenAITokenizer(
            tokenizer=encoding,
            max_tokens=chunk_max_tokens,
        )
        self.chunker = HybridChunker(tokenizer=tokenizer, merge_peers=True)

    def discover(self, target: Path) -> list[Path]:
        return discover_files(target)

    def ingest_file(
        self,
        path: Path,
        *,
        content_hash: str,
        index_fingerprint: str,
    ) -> list[DocumentChunk]:
        path = path.expanduser().resolve()
        result = self.converter.convert(path, raises_on_error=False)
        if self.strict_conversion and result.status.value != "success":
            raise RuntimeError(_conversion_error_message(path, result))

        line_mapper = (
            SourceLineMapper.from_path(path)
            if path.suffix.lower() in {".txt", ".md"}
            else None
        )
        source = str(path)
        chunks: list[DocumentChunk] = []
        for index, chunk in enumerate(
            self.chunker.chunk(dl_doc=result.document),
            start=1,
        ):
            text = chunk.text.strip()
            if not text:
                continue
            embedding_text = self.chunker.contextualize(chunk=chunk).strip()
            raw_meta = chunk.meta.export_json_dict()
            locator = extract_source_locator(
                path,
                raw_meta,
                index,
                text,
                line_mapper,
            )
            metadata = locator.to_metadata(
                content_hash=content_hash,
                index_fingerprint=index_fingerprint,
            )
            chunks.append(
                DocumentChunk(
                    id=_chunk_id(
                        source,
                        content_hash,
                        index_fingerprint,
                        index,
                        locator.element_refs,
                        text,
                    ),
                    text=text,
                    embedding_text=embedding_text,
                    source=source,
                    location=locator.location,
                    metadata=metadata,
                )
            )

        if not chunks:
            raise RuntimeError(f"文档已找到，但 Docling 没有产生可索引文本：{path}")
        for document_chunk in chunks:
            document_chunk.metadata["source_chunk_count"] = len(chunks)
        return chunks
