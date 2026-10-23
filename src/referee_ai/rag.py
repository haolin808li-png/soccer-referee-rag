"""Small, auditable rule retrieval layer for the referee assistant.

The default implementation is deliberately dependency free.  A dense retriever
and a reranker can be injected for production GPU runs without changing the
index format or the decision code.
"""

from __future__ import annotations

import json
import math
import re
from collections.abc import Iterable, Mapping, Sequence
from dataclasses import asdict, dataclass
from hashlib import sha256
from pathlib import Path
from typing import Any, Protocol

TOKEN_RE = re.compile(r"[\w-]+", re.UNICODE)
INJECTION_RE = re.compile(
    r"(ignore|disregard|override).{0,30}(instruction|prompt|system)|"
    r"(system|assistant|developer)\s*(message|prompt)|"
    r"do\s+not\s+follow|```(?:system|assistant|developer)",
    re.IGNORECASE,
)


def tokenize(text: str) -> list[str]:
    return [x.casefold() for x in TOKEN_RE.findall(text)]


def prompt_safe_rule_text(text: str) -> str:
    """Neutralize instruction-like text while retaining useful rule prose."""
    text = text.replace("```", "''' ")
    return INJECTION_RE.sub("[instruction-like text removed]", text)


@dataclass(frozen=True)
class RuleChunk:
    id: str
    text: str
    version: str
    law: str
    section: str
    concept: str
    sanction: str
    source: str = ""

    @classmethod
    def from_mapping(cls, row: Mapping[str, Any], source: str = "") -> RuleChunk:
        required = ("text", "version", "law", "section", "concept", "sanction")
        missing = [key for key in required if not str(row.get(key, "")).strip()]
        if missing:
            raise ValueError(f"rule chunk missing required fields: {', '.join(missing)}")
        canonical = "|".join(str(row[k]).strip() for k in required)
        chunk_id = str(row.get("id") or f"rule-{sha256(canonical.encode()).hexdigest()[:16]}")
        return cls(
            id=chunk_id,
            text=str(row["text"]).strip(),
            version=str(row["version"]).strip(),
            law=str(row["law"]).strip(),
            section=str(row["section"]).strip(),
            concept=str(row["concept"]).strip(),
            sanction=str(row["sanction"]).strip(),
            source=str(row.get("source") or source),
        )


@dataclass(frozen=True)
class SearchHit:
    chunk: RuleChunk
    score: float
    rank: int

    def citation(self) -> dict[str, str]:
        c = self.chunk
        return {
            "id": c.id,
            "version": c.version,
            "law": c.law,
            "section": c.section,
            "source": c.source,
        }


class DenseRetriever(Protocol):
    def search(self, query: str, chunks: Sequence[RuleChunk], top_k: int) -> Sequence[tuple[str, float]]: ...


class Reranker(Protocol):
    def __call__(self, query: str, hits: Sequence[SearchHit]) -> Sequence[SearchHit]: ...


class SentenceTransformerFaissRetriever:
    """Optional FAISS backend, imported only when selected in a GPU environment."""

    def __init__(self, model_name: str = "BAAI/bge-m3", device: str | None = None):
        try:
            import faiss  # type: ignore
            from sentence_transformers import SentenceTransformer  # type: ignore
        except ImportError as exc:  # pragma: no cover - exercised only with rag extras
            raise RuntimeError("install referee-ai[rag] to use dense FAISS retrieval") from exc
        self._faiss = faiss
        self._model = SentenceTransformer(model_name, device=device)
        self._ids: tuple[str, ...] = ()
        self._index: Any = None

    def _fit(self, chunks: Sequence[RuleChunk]) -> None:
        ids = tuple(c.id for c in chunks)
        if ids == self._ids:
            return
        texts = [f"{c.law} {c.section} {c.concept} {c.sanction} {c.text}" for c in chunks]
        vectors = self._model.encode(texts, normalize_embeddings=True, convert_to_numpy=True)
        vectors = vectors.astype("float32")
        self._index = self._faiss.IndexFlatIP(vectors.shape[1])
        self._index.add(vectors)
        self._ids = ids

    def search(self, query: str, chunks: Sequence[RuleChunk], top_k: int) -> Sequence[tuple[str, float]]:
        self._fit(chunks)
        vector = self._model.encode([query], normalize_embeddings=True, convert_to_numpy=True)
        scores, indices = self._index.search(vector.astype("float32"), min(top_k, len(chunks)))
        return [(self._ids[int(idx)], float(score)) for idx, score in zip(indices[0], scores[0]) if idx >= 0]


def _frontmatter(text: str) -> tuple[dict[str, str], str]:
    if not text.startswith("---\n"):
        return {}, text
    end = text.find("\n---\n", 4)
    if end < 0:
        return {}, text
    meta: dict[str, str] = {}
    for line in text[4:end].splitlines():
        if ":" in line:
            key, value = line.split(":", 1)
            meta[key.strip()] = value.strip().strip('"\'')
    return meta, text[end + 5 :]


def load_rule_documents(paths: Iterable[str | Path]) -> list[RuleChunk]:
    """Load JSONL rows or Markdown sections with YAML-like frontmatter.

    Markdown frontmatter supplies version/law and defaults; every ``##`` section
    becomes a chunk. Per-section metadata can be written on one HTML comment,
    e.g. ``<!-- concept: reckless; sanction: yellow_card -->``.
    """
    chunks: list[RuleChunk] = []
    for raw_path in paths:
        path = Path(raw_path)
        if path.suffix.lower() == ".jsonl":
            for line_no, line in enumerate(path.read_text(encoding="utf-8").splitlines(), 1):
                if not line.strip():
                    continue
                try:
                    row = json.loads(line)
                except json.JSONDecodeError as exc:
                    raise ValueError(f"{path}:{line_no}: invalid JSON") from exc
                chunks.append(RuleChunk.from_mapping(row, str(path)))
            continue
        if path.suffix.lower() not in {".md", ".markdown"}:
            raise ValueError(f"unsupported rule document: {path}")
        meta, body = _frontmatter(path.read_text(encoding="utf-8"))
        sections = re.split(r"(?m)^##\s+", body)
        for part in sections[1:] if len(sections) > 1 else sections:
            lines = part.strip().splitlines()
            if not lines:
                continue
            section = lines.pop(0).strip().rstrip("#").strip()
            local = dict(meta)
            if lines and re.fullmatch(r"<!--.*-->", lines[0].strip()):
                annotation = lines.pop(0).strip()[4:-3]
                for item in annotation.split(";"):
                    if ":" in item:
                        key, value = item.split(":", 1)
                        local[key.strip()] = value.strip()
            row = {
                "text": "\n".join(lines).strip(),
                "version": local.get("version", "unknown"),
                "law": local.get("law", path.stem),
                "section": section or local.get("section", path.stem),
                "concept": local.get("concept", section),
                "sanction": local.get("sanction", "context_dependent"),
                "source": str(path),
            }
            chunks.append(RuleChunk.from_mapping(row, str(path)))
    if len({chunk.id for chunk in chunks}) != len(chunks):
        raise ValueError("duplicate rule chunk id")
    return chunks


class RuleIndex:
    def __init__(self, chunks: Sequence[RuleChunk], dense: DenseRetriever | None = None):
        if not chunks:
            raise ValueError("at least one rule chunk is required")
        self.chunks = list(chunks)
        self.dense = dense
        self._terms = [tokenize(" ".join((c.text, c.law, c.section, c.concept, c.sanction))) for c in chunks]
        self._df: dict[str, int] = {}
        for terms in self._terms:
            for term in set(terms):
                self._df[term] = self._df.get(term, 0) + 1
        self._avgdl = sum(map(len, self._terms)) / len(self._terms)

    def _bm25(self, query: str) -> dict[str, float]:
        q = tokenize(query)
        n, k1, b = len(self.chunks), 1.5, 0.75
        scores: dict[str, float] = {}
        for chunk, terms in zip(self.chunks, self._terms):
            counts = {term: terms.count(term) for term in set(q)}
            score = 0.0
            for term in q:
                tf = counts.get(term, 0)
                if not tf:
                    continue
                df = self._df.get(term, 0)
                idf = math.log(1 + (n - df + 0.5) / (df + 0.5))
                denom = tf + k1 * (1 - b + b * len(terms) / max(self._avgdl, 1))
                score += idf * tf * (k1 + 1) / denom
            scores[chunk.id] = score
        return scores

    def search(self, query: str, top_k: int = 5, reranker: Reranker | None = None) -> list[SearchHit]:
        if top_k < 1:
            raise ValueError("top_k must be positive")
        sparse = self._bm25(query)
        combined = dict(sparse)
        if self.dense:
            dense_hits = self.dense.search(query, self.chunks, max(top_k * 4, 20))
            # Weighted reciprocal-rank fusion is robust to incomparable score scales.
            for rank, (chunk_id, _) in enumerate(dense_hits, 1):
                combined[chunk_id] = combined.get(chunk_id, 0.0) + 2.0 / (60 + rank)
        by_id = {c.id: c for c in self.chunks}
        ordered = sorted(combined.items(), key=lambda item: (-item[1], item[0]))[:top_k]
        hits = [SearchHit(by_id[cid], float(score), rank) for rank, (cid, score) in enumerate(ordered, 1)]
        if reranker:
            reranked = list(reranker(query, hits))[:top_k]
            hits = [SearchHit(hit.chunk, hit.score, rank) for rank, hit in enumerate(reranked, 1)]
        return hits

    def save(self, path: str | Path) -> None:
        payload = {"format": "referee-rule-index-v1", "chunks": [asdict(c) for c in self.chunks]}
        target = Path(path)
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")

    @classmethod
    def load(cls, path: str | Path, dense: DenseRetriever | None = None) -> RuleIndex:
        payload = json.loads(Path(path).read_text(encoding="utf-8"))
        if payload.get("format") != "referee-rule-index-v1":
            raise ValueError("unsupported rule index format")
        return cls([RuleChunk(**row) for row in payload["chunks"]], dense=dense)
