from __future__ import annotations

import json
import re
from pathlib import Path
from typing import Any, Iterable

from .models import SourceLocator

_REF_PATTERN = re.compile(r"^#/(?P<kind>[^/]+)/(?P<index>\d+)$")
_REF_LABELS = {
    "texts": "文本项",
    "tables": "表格",
    "pictures": "图片",
    "key_value_items": "键值项",
}


def _normalized(value: str) -> str:
    return "".join(character.casefold() for character in value if character.isalnum())


class SourceLineMapper:
    """Map parsed element text back to stable lines in a plain-text source."""

    def __init__(self, text: str) -> None:
        characters: list[str] = []
        line_numbers: list[int] = []
        for line_number, line in enumerate(text.splitlines(keepends=True), start=1):
            for character in line:
                if not character.isalnum():
                    continue
                folded = character.casefold()
                characters.extend(folded)
                line_numbers.extend([line_number] * len(folded))
        self._text = "".join(characters)
        self._line_numbers = line_numbers
        self._cursor = 0

    @classmethod
    def from_path(cls, path: Path) -> SourceLineMapper | None:
        raw = path.read_bytes()
        for encoding in ("utf-8-sig", "utf-8", "gb18030"):
            try:
                return cls(raw.decode(encoding))
            except UnicodeDecodeError:
                continue
        return None

    def locate(
        self,
        element_texts: Iterable[str],
        fallback_text: str,
    ) -> tuple[int, int] | None:
        candidates = [value for value in element_texts if _normalized(value)]
        if not candidates:
            candidates = [fallback_text]

        total_characters = sum(len(_normalized(value)) for value in candidates)
        matched_characters = 0
        matched_ranges: list[tuple[int, int]] = []
        cursor = self._cursor

        for value in candidates:
            normalized = _normalized(value)
            if not normalized:
                continue
            start = self._text.find(normalized, cursor)
            if start < 0:
                continue
            end = start + len(normalized)
            matched_characters += len(normalized)
            matched_ranges.append((start, end))
            cursor = end

        if (
            not matched_ranges
            or matched_characters < 6
            or matched_characters / max(total_characters, 1) < 0.6
        ):
            return None

        start = min(item[0] for item in matched_ranges)
        end = max(item[1] for item in matched_ranges)
        if end <= start or end > len(self._line_numbers):
            return None
        self._cursor = end
        return self._line_numbers[start], self._line_numbers[end - 1]


def _compact_element_labels(refs: Iterable[str]) -> tuple[str, ...]:
    grouped: dict[str, list[int]] = {}
    for ref in refs:
        match = _REF_PATTERN.match(ref)
        if not match:
            continue
        kind = match.group("kind")
        if kind not in _REF_LABELS:
            continue
        grouped.setdefault(kind, []).append(int(match.group("index")) + 1)

    labels: list[str] = []
    for kind, indexes in grouped.items():
        ordered = sorted(set(indexes))
        start = previous = ordered[0]
        for index in ordered[1:] + [ordered[-1] + 2]:
            if index == previous + 1:
                previous = index
                continue
            number = str(start) if start == previous else f"{start}–{previous}"
            labels.append(f"{_REF_LABELS[kind]} {number}")
            start = previous = index
    return tuple(labels)


def _headings(meta: dict[str, Any]) -> tuple[str, ...]:
    values: list[str] = []
    for value in meta.get("headings") or []:
        heading = str(value).strip()
        if heading and (not values or values[-1] != heading):
            values.append(heading)
    return tuple(values)


def extract_source_locator(
    path: Path,
    meta: dict[str, Any],
    chunk_index: int,
    chunk_text: str,
    line_mapper: SourceLineMapper | None = None,
) -> SourceLocator:
    pages: set[int] = set()
    refs: list[str] = []
    element_texts: list[str] = []
    provenance: list[dict[str, Any]] = []

    for raw_item in meta.get("doc_items") or []:
        if not isinstance(raw_item, dict):
            continue
        ref = str(raw_item.get("self_ref") or "").strip()
        label = str(raw_item.get("label") or "").strip()
        if ref:
            refs.append(ref)
        text = raw_item.get("text")
        if isinstance(text, str) and text.strip():
            element_texts.append(text)

        for raw_prov in raw_item.get("prov") or []:
            if not isinstance(raw_prov, dict):
                continue
            page_no = raw_prov.get("page_no")
            if isinstance(page_no, int):
                pages.add(page_no)
            provenance.append(
                {
                    "ref": ref,
                    "label": label,
                    "page_no": page_no,
                    "bbox": raw_prov.get("bbox"),
                    "charspan": raw_prov.get("charspan"),
                }
            )

    suffix = path.suffix.lower().lstrip(".")
    page_start = min(pages) if pages and suffix == "pdf" else None
    page_end = max(pages) if pages and suffix == "pdf" else None
    line_range = (
        line_mapper.locate(element_texts, chunk_text)
        if line_mapper and suffix in {"txt", "md"}
        else None
    )

    return SourceLocator(
        source=str(path.resolve()),
        source_name=path.name,
        file_type=suffix,
        chunk_index=chunk_index,
        heading_path=_headings(meta),
        page_start=page_start,
        page_end=page_end,
        line_start=line_range[0] if line_range else None,
        line_end=line_range[1] if line_range else None,
        element_refs=tuple(dict.fromkeys(refs)),
        element_labels=_compact_element_labels(refs),
        provenance_json=json.dumps(
            provenance,
            ensure_ascii=False,
            separators=(",", ":"),
        ),
    )
