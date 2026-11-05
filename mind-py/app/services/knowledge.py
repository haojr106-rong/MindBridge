from __future__ import annotations

import heapq
import json
import logging
import math
import re
import threading
import weakref
from dataclasses import dataclass
from datetime import datetime
from typing import Hashable, Iterable

from pypdf import PdfReader
from sqlalchemy import func
from sqlalchemy.engine import Connection, Engine
from sqlalchemy.orm import Session

from app.core.config import Settings
from app.models.entities import KnowledgeChunk
from app.services.search_store import OpenSearchKnowledgeStore
from app.services.vector_store import FALLBACK_RETRIEVAL_LABEL, PRIMARY_RETRIEVAL_LABEL, ChromaKnowledgeStore


logger = logging.getLogger(__name__)


@dataclass
class SearchResult:
    chunk_id: int | None
    source: str
    content: str
    score: float


@dataclass
class RetrievalCandidate:
    result: SearchResult
    vector_score: float = 0.0
    bm25_score: float = 0.0


@dataclass(frozen=True)
class KnowledgeCorpusSignature:
    """Cheap corpus version used to invalidate process-local search indexes."""

    chunk_count: int
    max_chunk_id: int
    latest_created_at: datetime | None


class Bm25InvertedIndex:
    """Thread-safe BM25 index shared by services using one database engine.

    Tokenization, document lengths and document frequencies are computed when
    the corpus changes. Queries only visit postings for tokens in the query.
    """

    def __init__(self):
        self._lock = threading.RLock()
        self._document_terms: dict[int, tuple[str, ...]] = {}
        self._document_lengths: dict[int, int] = {}
        self._document_sources: dict[int, str] = {}
        self._source_documents: dict[str, set[int]] = {}
        self._postings: dict[str, dict[int, int]] = {}
        self._total_document_length = 0
        self._signature: KnowledgeCorpusSignature | None = None

    @property
    def ready(self) -> bool:
        with self._lock:
            return self._signature is not None

    def matches(self, signature: KnowledgeCorpusSignature) -> bool:
        with self._lock:
            return self._signature == signature

    def rebuild(
        self,
        chunks: Iterable[KnowledgeChunk],
        signature: KnowledgeCorpusSignature,
    ) -> None:
        with self._lock:
            self._document_terms.clear()
            self._document_lengths.clear()
            self._document_sources.clear()
            self._source_documents.clear()
            self._postings.clear()
            self._total_document_length = 0
            for chunk in chunks:
                self._add_chunk(chunk)
            self._signature = signature

    def replace_source(
        self,
        source: str,
        chunks: Iterable[KnowledgeChunk],
        signature: KnowledgeCorpusSignature,
    ) -> None:
        with self._lock:
            for chunk_id in tuple(self._source_documents.get(source, ())):
                self._remove_chunk(chunk_id)
            for chunk in chunks:
                self._add_chunk(chunk)
            self._signature = signature

    def search(self, query: str, top_k: int) -> list[tuple[int, float]]:
        query_terms = counts(tokenize(query))
        if not query_terms or top_k <= 0:
            return []
        with self._lock:
            total_documents = len(self._document_lengths)
            if total_documents == 0:
                return []
            average_length = self._total_document_length / total_documents or 1.0
            candidate_ids: set[int] = set()
            term_statistics: list[tuple[dict[int, int], float, float]] = []
            for term, query_frequency in query_terms.items():
                posting = self._postings.get(term)
                if not posting:
                    continue
                candidate_ids.update(posting)
                document_frequency = len(posting)
                inverse_document_frequency = math.log(
                    1.0
                    + (total_documents - document_frequency + 0.5)
                    / (document_frequency + 0.5)
                )
                term_statistics.append(
                    (posting, inverse_document_frequency, 1.0 + math.log(query_frequency))
                )

            k1 = 1.5
            b = 0.75
            scores: list[tuple[int, float]] = []
            for chunk_id in candidate_ids:
                document_length = self._document_lengths[chunk_id]
                length_norm = k1 * (1.0 - b + b * document_length / average_length)
                score = 0.0
                for posting, inverse_document_frequency, query_boost in term_statistics:
                    term_frequency = posting.get(chunk_id, 0)
                    if term_frequency == 0:
                        continue
                    score += (
                        inverse_document_frequency
                        * query_boost
                        * (term_frequency * (k1 + 1.0))
                        / (term_frequency + length_norm)
                    )
                if score > 0:
                    scores.append((chunk_id, score))
            return heapq.nlargest(top_k, scores, key=lambda item: (item[1], -item[0]))

    def statistics(self) -> dict[str, int | float | bool]:
        with self._lock:
            document_count = len(self._document_lengths)
            return {
                "ready": self._signature is not None,
                "documents": document_count,
                "terms": len(self._postings),
                "postings": sum(len(posting) for posting in self._postings.values()),
                "averageDocumentLength": round(
                    self._total_document_length / document_count if document_count else 0.0,
                    3,
                ),
            }

    def _add_chunk(self, chunk: KnowledgeChunk) -> None:
        if chunk.id is None:
            return
        chunk_id = int(chunk.id)
        if chunk_id in self._document_lengths:
            self._remove_chunk(chunk_id)
        term_frequencies = counts(tokenize(chunk.content))
        document_length = sum(term_frequencies.values())
        self._document_terms[chunk_id] = tuple(term_frequencies)
        self._document_lengths[chunk_id] = document_length
        self._document_sources[chunk_id] = chunk.source
        self._source_documents.setdefault(chunk.source, set()).add(chunk_id)
        self._total_document_length += document_length
        for term, frequency in term_frequencies.items():
            self._postings.setdefault(term, {})[chunk_id] = frequency

    def _remove_chunk(self, chunk_id: int) -> None:
        document_terms = self._document_terms.pop(chunk_id, ())
        self._total_document_length -= self._document_lengths.pop(chunk_id, 0)
        source = self._document_sources.pop(chunk_id, "")
        source_documents = self._source_documents.get(source)
        if source_documents is not None:
            source_documents.discard(chunk_id)
            if not source_documents:
                self._source_documents.pop(source, None)
        for term in document_terms:
            posting = self._postings.get(term)
            if posting is None:
                continue
            posting.pop(chunk_id, None)
            if not posting:
                self._postings.pop(term, None)


_BM25_REGISTRY_LOCK = threading.RLock()
_BM25_INDEXES: "weakref.WeakKeyDictionary[Engine, Bm25InvertedIndex]" = weakref.WeakKeyDictionary()


def _engine_for_session(db: Session) -> Engine:
    bind = db.get_bind()
    return bind.engine if isinstance(bind, Connection) else bind


def _shared_bm25_index(db: Session) -> Bm25InvertedIndex:
    engine = _engine_for_session(db)
    with _BM25_REGISTRY_LOCK:
        index = _BM25_INDEXES.get(engine)
        if index is None:
            index = Bm25InvertedIndex()
            _BM25_INDEXES[engine] = index
        return index


class KnowledgeService:
    def __init__(self, db: Session, settings: Settings):
        self.db = db
        self.settings = settings
        self.vector_store = ChromaKnowledgeStore(settings)
        self.search_store = OpenSearchKnowledgeStore(settings)
        self.bm25_index = _shared_bm25_index(db)

    def count(self) -> int:
        return self.db.query(KnowledgeChunk).count()

    def ensure_source(self, source: str, content: str) -> int:
        chunks = chunk_text(content, self.settings.knowledge_chunk_size, self.settings.knowledge_chunk_overlap)
        existing = [
            chunk.content
            for chunk in self.db.query(KnowledgeChunk)
            .filter(KnowledgeChunk.source == source)
            .order_by(KnowledgeChunk.source_index.asc())
            .all()
        ]
        if existing == chunks:
            self.warm_bm25_index()
            return len(existing)
        return self.ingest(source, content)

    def status(self) -> dict:
        vector_chunks = None
        vector_error = getattr(self.vector_store, "error", "")
        if self.vector_store.can_embed:
            try:
                vector_chunks = self.vector_store.count()
            except Exception as exc:
                vector_error = f"{type(exc).__name__}: {exc}"
        return {
            "retrievalOrder": [
                PRIMARY_RETRIEVAL_LABEL,
                "OpenSearch BM25 keyword retrieval",
                f"{FALLBACK_RETRIEVAL_LABEL} when vector/OpenSearch retrieval is unavailable",
            ],
            "primaryRetrieval": PRIMARY_RETRIEVAL_LABEL,
            "fallbackRetrieval": FALLBACK_RETRIEVAL_LABEL,
            "databaseChunks": self.count(),
            "vectorEnabled": self.settings.knowledge_vector_enabled,
            "vectorAvailable": self.vector_store.can_embed,
            "vectorRequired": self.settings.knowledge_vector_required,
            "embeddingModel": self.settings.openai_embedding_model,
            "vectorChunks": vector_chunks,
            "chromaPersistDir": self.settings.chroma_persist_dir,
            "chromaCollectionName": self.settings.chroma_collection_name,
            "chromaSnapshotDir": self.settings.chroma_snapshot_dir,
            "candidateK": self.settings.knowledge_candidate_k,
            "hybridVectorWeight": self.settings.knowledge_hybrid_vector_weight,
            "hybridBm25Weight": self.settings.knowledge_hybrid_bm25_weight,
            "rerankEnabled": self.settings.knowledge_rerank_enabled,
            "keywordSearch": self.search_store.statistics(),
            "bm25Index": self.bm25_index.statistics(),
            "vectorError": vector_error,
        }

    def rebuild_vector_index(self) -> int:
        if not self.vector_store.can_embed:
            raise RuntimeError(getattr(self.vector_store, "error", "") or "Chroma 向量库不可用")
        rows = self.db.query(KnowledgeChunk).order_by(KnowledgeChunk.source.asc(), KnowledgeChunk.source_index.asc()).all()
        self._sync_vector_chunks(rows)
        self.db.commit()
        return len(rows)

    def backup_vector_index(self) -> str:
        if not self.vector_store.can_embed:
            raise RuntimeError(getattr(self.vector_store, "error", "") or "Chroma 向量库不可用")
        snapshot = self.vector_store.snapshot()
        if snapshot is None:
            raise RuntimeError("Chroma 持久化目录不存在，无法生成快照")
        return snapshot

    def ingest(self, source: str, content: str) -> int:
        previous_signature = self._bm25_signature()
        chunks = chunk_text(content, self.settings.knowledge_chunk_size, self.settings.knowledge_chunk_overlap)
        self._delete_vector_source(source)
        self.db.query(KnowledgeChunk).filter(KnowledgeChunk.source == source).delete()
        rows = []
        for index, chunk in enumerate(chunks):
            row = KnowledgeChunk(source=source, source_index=index, content=chunk)
            self.db.add(row)
            rows.append(row)
        self.db.flush()
        self._index_vector_chunks(rows)
        self.db.commit()
        committed_rows = (
            self.db.query(KnowledgeChunk)
            .filter(KnowledgeChunk.source == source)
            .order_by(KnowledgeChunk.source_index.asc())
            .all()
        )
        signature = self._bm25_signature()
        if self.bm25_index.matches(previous_signature):
            self.bm25_index.replace_source(source, committed_rows, signature)
        else:
            self._rebuild_bm25_index(signature)
        self._sync_search_source(source, committed_rows)
        return len(chunks)

    def warm_bm25_index(self) -> dict[str, int | float | bool]:
        """Build or refresh the process-local BM25 index when the corpus changed."""
        self._ensure_bm25_index()
        return self.bm25_index.statistics()

    def ensure_search_index(self) -> dict[str, object]:
        """Create/rebuild the persistent BM25 projection when it is stale."""
        if not self.search_store.enabled:
            return self.search_store.statistics()
        try:
            if not self.search_store.available:
                raise RuntimeError(self.search_store.error or "OpenSearch is unavailable")
            database_signature = self._bm25_signature()
            search_signature = self.search_store.corpus_signature()
            expected = (database_signature.chunk_count, database_signature.max_chunk_id)
            if search_signature != expected:
                rows = self.db.query(KnowledgeChunk).order_by(KnowledgeChunk.id.asc()).all()
                self.search_store.rebuild(rows)
            return self.search_store.statistics()
        except Exception as exc:
            self._handle_search_error("ensure_index", exc)
            return self.search_store.statistics()

    def rebuild_search_index(self) -> int:
        rows = self.db.query(KnowledgeChunk).order_by(KnowledgeChunk.id.asc()).all()
        try:
            return self.search_store.rebuild(rows)
        except Exception as exc:
            self._handle_search_error("rebuild", exc)
            return 0

    def ingest_file(self, filename: str, data: bytes) -> int:
        lower = filename.lower()
        if lower.endswith(".pdf"):
            text = extract_pdf(data)
        else:
            text = data.decode("utf-8", errors="ignore")
        return self.ingest(filename, text)

    def retrieve(self, query: str, top_k: int | None = None) -> list[SearchResult]:
        top_k = top_k or self.settings.knowledge_top_k
        candidate_k = self._candidate_k(top_k)
        # Primary retrieval now uses hybrid recall: semantic vector candidates
        # plus BM25 keyword candidates, followed by deterministic local rerank.
        vector_results = self._retrieve_vector(query, candidate_k)
        bm25_results = self._retrieve_bm25(query, candidate_k)
        ranked = self._fuse_and_rerank(query, vector_results, bm25_results, top_k)
        if ranked:
            return self._expand_best(ranked, top_k)
        return []

    def _retrieve_bm25(self, query: str, top_k: int, chunks: list[KnowledgeChunk] | None = None) -> list[SearchResult]:
        # Preserve the explicit chunk-list path for isolated one-shot callers.
        if chunks is not None:
            scores = bm25_scores(query, chunks)
            ranked = [
                SearchResult(chunk.id, chunk.source, chunk.content, scores.get(chunk.id, 0.0))
                for chunk in chunks
                if chunk.id is not None and scores.get(chunk.id, 0.0) > 0
            ]
            ranked.sort(key=lambda item: item.score, reverse=True)
            return ranked[:top_k]

        if self.search_store.enabled:
            try:
                return [
                    SearchResult(hit.chunk_id, hit.source, hit.content, hit.score)
                    for hit in self.search_store.search(query, top_k)
                ]
            except Exception as exc:
                self._handle_search_error("retrieve", exc)

        self._ensure_bm25_index()
        scored_ids = self.bm25_index.search(query, top_k)
        if not scored_ids:
            return []
        chunk_ids = [chunk_id for chunk_id, _ in scored_ids]
        rows = self.db.query(KnowledgeChunk).filter(KnowledgeChunk.id.in_(chunk_ids)).all()
        chunks_by_id = {int(chunk.id): chunk for chunk in rows if chunk.id is not None}
        return [
            SearchResult(chunk_id, chunks_by_id[chunk_id].source, chunks_by_id[chunk_id].content, score)
            for chunk_id, score in scored_ids
            if chunk_id in chunks_by_id
        ]

    def _ensure_bm25_index(self) -> None:
        signature = self._bm25_signature()
        if self.bm25_index.matches(signature):
            return
        self._rebuild_bm25_index(signature)

    def _rebuild_bm25_index(self, signature: KnowledgeCorpusSignature | None = None) -> None:
        chunks = self.db.query(KnowledgeChunk).order_by(KnowledgeChunk.id.asc()).all()
        self.bm25_index.rebuild(chunks, signature or self._bm25_signature())

    def _bm25_signature(self) -> KnowledgeCorpusSignature:
        chunk_count, max_chunk_id, latest_created_at = self.db.query(
            func.count(KnowledgeChunk.id),
            func.max(KnowledgeChunk.id),
            func.max(KnowledgeChunk.created_at),
        ).one()
        return KnowledgeCorpusSignature(
            chunk_count=int(chunk_count or 0),
            max_chunk_id=int(max_chunk_id or 0),
            latest_created_at=latest_created_at,
        )

    def _fuse_and_rerank(
        self,
        query: str,
        vector_results: list[SearchResult],
        bm25_results: list[SearchResult],
        top_k: int,
    ) -> list[SearchResult]:
        candidates: dict[Hashable, RetrievalCandidate] = {}
        vector_scores = {result_key(item): item.score for item in vector_results if item.score > 0}
        bm25_scores_by_key = {result_key(item): item.score for item in bm25_results if item.score > 0}
        normalized_vector = normalize_scores(vector_scores)
        normalized_bm25 = normalize_scores(bm25_scores_by_key)

        for item in [*vector_results, *bm25_results]:
            key = result_key(item)
            candidate = candidates.get(key)
            if candidate is None:
                candidate = RetrievalCandidate(result=item)
                candidates[key] = candidate
            candidate.vector_score = max(candidate.vector_score, normalized_vector.get(key, 0.0))
            candidate.bm25_score = max(candidate.bm25_score, normalized_bm25.get(key, 0.0))

        if not candidates:
            return []

        vector_weight = max(0.0, self.settings.knowledge_hybrid_vector_weight) if vector_results else 0.0
        bm25_weight = max(0.0, self.settings.knowledge_hybrid_bm25_weight)
        if vector_weight == 0.0 and bm25_weight == 0.0:
            bm25_weight = 1.0
        total_weight = vector_weight + bm25_weight

        fused = []
        for candidate in candidates.values():
            score = (
                candidate.vector_score * vector_weight
                + candidate.bm25_score * bm25_weight
            ) / total_weight
            fused.append(replace_score(candidate.result, score))

        fused.sort(key=lambda item: item.score, reverse=True)
        fused = fused[:self._candidate_k(top_k)]
        return self._rerank(query, fused, top_k)

    def _rerank(self, query: str, candidates: list[SearchResult], top_k: int) -> list[SearchResult]:
        if not self.settings.knowledge_rerank_enabled:
            return candidates[:top_k]
        reranked = [
            replace_score(item, rerank_score(query, item.content, item.score))
            for item in candidates
        ]
        reranked.sort(key=lambda item: item.score, reverse=True)
        return reranked[:top_k]

    def _candidate_k(self, top_k: int) -> int:
        return max(top_k, self.settings.knowledge_candidate_k)

    def _retrieve_vector(self, query: str, top_k: int) -> list[SearchResult]:
        if not self.vector_store.can_embed:
            return []
        try:
            self._ensure_vector_index()
            query_embedding = self.vector_store.embed_texts([query])[0]
            hits = self.vector_store.query(query_embedding, top_k)
        except Exception as exc:
            self._handle_vector_error("retrieve", exc)
            return []
        results = []
        for hit in hits:
            chunk = self.db.get(KnowledgeChunk, hit.chunk_id) if hit.chunk_id is not None else None
            results.append(
                SearchResult(
                    chunk.id if chunk is not None else hit.chunk_id,
                    chunk.source if chunk is not None else hit.source,
                    chunk.content if chunk is not None else hit.content,
                    hit.score,
                )
            )
        return results

    def _ensure_vector_index(self) -> None:
        rows = self.db.query(KnowledgeChunk).order_by(KnowledgeChunk.source.asc(), KnowledgeChunk.source_index.asc()).all()
        if not rows:
            return
        if (
            self.vector_store.count() == len(rows)
            and all(row.embedding_json for row in rows)
            and self.vector_store.has_exact_chunk_ids(rows)
        ):
            return
        self._sync_vector_chunks(rows)
        self.db.commit()

    def _delete_vector_source(self, source: str) -> None:
        if not self.vector_store.can_embed:
            return
        try:
            self.vector_store.delete_source(source)
        except Exception as exc:
            self._handle_vector_error("delete_source", exc)

    def _index_vector_chunks(self, chunks: list[KnowledgeChunk]) -> None:
        if not chunks or not self.vector_store.can_embed:
            return
        try:
            embeddings = self._embeddings_for_chunks(chunks)
            for chunk, embedding in zip(chunks, embeddings):
                chunk.embedding_json = json.dumps(embedding, separators=(",", ":"))
            self.vector_store.upsert_chunks(chunks, embeddings)
        except Exception as exc:
            self._handle_vector_error("index", exc)

    def _sync_vector_chunks(self, chunks: list[KnowledgeChunk]) -> None:
        if not chunks or not self.vector_store.can_embed:
            return
        try:
            embeddings = self._embeddings_for_chunks(chunks)
            for chunk, embedding in zip(chunks, embeddings):
                chunk.embedding_json = json.dumps(embedding, separators=(",", ":"))
            self.vector_store.sync_chunks(chunks, embeddings)
        except Exception as exc:
            self._handle_vector_error("sync", exc)

    def _embeddings_for_chunks(self, chunks: list[KnowledgeChunk]) -> list[list[float]]:
        embeddings: list[list[float] | None] = []
        missing_indexes = []
        missing_texts = []
        for index, chunk in enumerate(chunks):
            embedding = parse_embedding(chunk.embedding_json)
            embeddings.append(embedding)
            if embedding is None:
                missing_indexes.append(index)
                missing_texts.append(chunk.content)
        if missing_texts:
            new_embeddings = self.vector_store.embed_texts(missing_texts)
            for index, embedding in zip(missing_indexes, new_embeddings):
                embeddings[index] = embedding
        resolved = [embedding for embedding in embeddings if embedding is not None]
        if len(resolved) != len(chunks):
            raise ValueError("Embedding response count did not match knowledge chunks.")
        return resolved

    def _handle_vector_error(self, action: str, exc: Exception) -> None:
        if self.settings.knowledge_vector_required:
            raise exc
        logger.warning(
            "%s %s failed; falling back to %s: %s",
            PRIMARY_RETRIEVAL_LABEL,
            action,
            FALLBACK_RETRIEVAL_LABEL,
            exc,
        )

    def _sync_search_source(self, source: str, chunks: list[KnowledgeChunk]) -> None:
        if not self.search_store.enabled:
            return
        try:
            self.search_store.replace_source(source, chunks)
        except Exception as exc:
            self._handle_search_error("replace_source", exc)

    def _handle_search_error(self, action: str, exc: Exception) -> None:
        self.search_store.error = f"{type(exc).__name__}: {exc}"
        if self.search_store.required:
            raise exc
        # Latch the failure for this request/service instance. A newly created
        # service will retry, while repeated bootstrap operations fall back
        # immediately instead of waiting for the same unavailable endpoint.
        self.search_store.client = None
        logger.warning(
            "OpenSearch BM25 %s failed; falling back to process-local BM25: %s",
            action,
            exc,
        )

    def _expand_best(self, ranked: list[SearchResult], top_k: int) -> list[SearchResult]:
        if not ranked:
            return []
        best = ranked[0]
        expanded = self._expand(best)
        results = [expanded]
        for item in ranked[1:]:
            if item.chunk_id != expanded.chunk_id and len(results) < top_k:
                results.append(item)
        return results

    def _expand(self, result: SearchResult) -> SearchResult:
        if result.chunk_id is None:
            return result
        chunk = self.db.get(KnowledgeChunk, result.chunk_id)
        if chunk is None:
            return result
        neighbors = (
            self.db.query(KnowledgeChunk)
            .filter(KnowledgeChunk.source == chunk.source)
            .filter(KnowledgeChunk.source_index >= max(0, chunk.source_index - 1))
            .filter(KnowledgeChunk.source_index <= chunk.source_index + 1)
            .order_by(KnowledgeChunk.source_index.asc())
            .all()
        )
        return SearchResult(chunk.id, chunk.source, "\n\n".join(item.content for item in neighbors), result.score)


def chunk_text(content: str, size: int, overlap: int) -> list[str]:
    text = re.sub(r"\s+", " ", content or "").strip()
    if not text:
        return []
    chunks = []
    start = 0
    step = max(1, size - overlap)
    while start < len(text):
        chunks.append(text[start:start + size])
        start += step
    return chunks


def hybrid_score(query: str, content: str) -> float:
    return token_cosine(query, content) * 0.75 + keyword_score(query, content) * 0.25


def bm25_scores(query: str, chunks: list[KnowledgeChunk]) -> dict[int, float]:
    query_terms = counts(tokenize(query))
    if not query_terms or not chunks:
        return {}

    documents = []
    doc_freqs: dict[str, int] = {}
    for chunk in chunks:
        if chunk.id is None:
            continue
        token_counts = counts(tokenize(chunk.content))
        documents.append((chunk.id, token_counts, sum(token_counts.values())))
        for term in token_counts:
            doc_freqs[term] = doc_freqs.get(term, 0) + 1

    total_docs = len(documents)
    if total_docs == 0:
        return {}
    average_length = sum(length for _, _, length in documents) / total_docs or 1.0
    k1 = 1.5
    b = 0.75
    scores: dict[int, float] = {}

    for chunk_id, token_counts, doc_length in documents:
        score = 0.0
        length_norm = k1 * (1.0 - b + b * doc_length / average_length)
        for term, query_frequency in query_terms.items():
            term_frequency = token_counts.get(term, 0)
            if term_frequency == 0:
                continue
            doc_frequency = doc_freqs.get(term, 0)
            idf = math.log(1.0 + (total_docs - doc_frequency + 0.5) / (doc_frequency + 0.5))
            query_boost = 1.0 + math.log(query_frequency)
            score += idf * query_boost * (term_frequency * (k1 + 1.0)) / (term_frequency + length_norm)
        if score > 0:
            scores[chunk_id] = score
    return scores


def rerank_score(query: str, content: str, base_score: float) -> float:
    lexical = hybrid_score(query, content)
    coverage = query_token_coverage(query, content)
    phrase = phrase_score(query, content)
    return base_score * 0.55 + lexical * 0.25 + coverage * 0.15 + phrase * 0.05


def query_token_coverage(query: str, content: str) -> float:
    query_tokens = set(tokenize(query))
    if not query_tokens:
        return 0.0
    content_tokens = set(tokenize(content))
    return len(query_tokens & content_tokens) / len(query_tokens)


def phrase_score(query: str, content: str) -> float:
    normalized_query = compact_text(query)
    if not normalized_query:
        return 0.0
    normalized_content = compact_text(content)
    if normalized_query in normalized_content:
        return 1.0
    return keyword_score(query, content)


def compact_text(text: str) -> str:
    return re.sub(r"\s+", "", text.lower())


def normalize_scores(scores: dict[Hashable, float]) -> dict[Hashable, float]:
    positives = [score for score in scores.values() if score > 0]
    if not positives:
        return {key: 0.0 for key in scores}
    lowest = min(positives)
    highest = max(positives)
    if math.isclose(lowest, highest):
        return {key: 1.0 if score > 0 else 0.0 for key, score in scores.items()}
    return {
        key: (score - lowest) / (highest - lowest) if score > 0 else 0.0
        for key, score in scores.items()
    }


def result_key(result: SearchResult) -> Hashable:
    return result.chunk_id if result.chunk_id is not None else (result.source, result.content)


def replace_score(result: SearchResult, score: float) -> SearchResult:
    return SearchResult(result.chunk_id, result.source, result.content, score)


def parse_embedding(raw: str | None) -> list[float] | None:
    if not raw:
        return None
    try:
        data = json.loads(raw)
    except json.JSONDecodeError:
        return None
    if not isinstance(data, list) or not data:
        return None
    if not all(isinstance(item, (int, float)) for item in data):
        return None
    return [float(item) for item in data]


def tokenize(text: str) -> list[str]:
    words = re.findall(r"[a-zA-Z0-9_]+|[\u4e00-\u9fff]", text.lower())
    grams = words[:]
    compact = "".join(ch for ch in text.lower() if "\u4e00" <= ch <= "\u9fff")
    grams.extend(compact[i:i + 2] for i in range(max(0, len(compact) - 1)))
    return [item for item in grams if item.strip()]


def token_cosine(left: str, right: str) -> float:
    left_counts = counts(tokenize(left))
    right_counts = counts(tokenize(right))
    if not left_counts or not right_counts:
        return 0.0
    dot = sum(value * right_counts.get(key, 0) for key, value in left_counts.items())
    left_norm = math.sqrt(sum(value * value for value in left_counts.values()))
    right_norm = math.sqrt(sum(value * value for value in right_counts.values()))
    return 0.0 if left_norm == 0 or right_norm == 0 else dot / (left_norm * right_norm)


def keyword_score(query: str, content: str) -> float:
    terms = [term for term in re.split(r"[\s，。！？、；：,.!?;:]+", query.lower()) if len(term) >= 2]
    if not terms:
        return 0.0
    lower = content.lower()
    matched = sum(1 for term in terms if term in lower)
    return min(1.0, matched / len(terms))


def counts(values: list[str]) -> dict[str, int]:
    result: dict[str, int] = {}
    for value in values:
        result[value] = result.get(value, 0) + 1
    return result


def extract_pdf(data: bytes) -> str:
    from io import BytesIO

    reader = PdfReader(BytesIO(data))
    return "\n".join(page.extract_text() or "" for page in reader.pages)
