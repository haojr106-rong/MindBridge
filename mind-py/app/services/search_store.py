from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable
from urllib.parse import urlparse

from app.core.config import Settings
from app.models.entities import KnowledgeChunk

try:
    from opensearchpy import OpenSearch, helpers
except ImportError:  # Allows the application to retain its local BM25 fallback.
    OpenSearch = None
    helpers = None


@dataclass(frozen=True)
class LexicalHit:
    chunk_id: int | None
    source: str
    content: str
    score: float


class OpenSearchKnowledgeStore:
    """Persistent BM25 index for large knowledge corpora.

    MySQL remains the source of truth. OpenSearch is a rebuildable search
    projection; callers can fall back to the process-local BM25 index if this
    service is unavailable.
    """

    def __init__(self, settings: Settings):
        self.settings = settings
        self.enabled = bool(getattr(settings, "knowledge_search_enabled", False))
        self.required = bool(getattr(settings, "knowledge_search_required", False))
        self.index_name = str(getattr(settings, "opensearch_index", "mindbridge_knowledge_chunks"))
        self.error = ""
        self.client = None
        if not self.enabled:
            self.error = "OpenSearch keyword retrieval is disabled"
            return
        if OpenSearch is None:
            self.error = "opensearch-py is not installed"
            if self.required:
                raise RuntimeError(self.error)
            return
        try:
            self.client = self._create_client()
        except Exception as exc:
            self.error = f"{type(exc).__name__}: {exc}"
            if self.required:
                raise

    @property
    def available(self) -> bool:
        if self.client is None:
            return False
        try:
            available = bool(self.client.ping())
            if not available:
                self.error = "OpenSearch ping failed"
            return available
        except Exception as exc:
            self.error = f"{type(exc).__name__}: {exc}"
            return False

    def ensure_index(self) -> None:
        client = self._require_client()
        if client.indices.exists(index=self.index_name):
            return
        client.indices.create(
            index=self.index_name,
            body={
                "settings": {
                    "number_of_shards": 1,
                    "number_of_replicas": 0,
                },
                "mappings": {
                    "properties": {
                        "chunk_id": {"type": "long"},
                        "source": {"type": "keyword"},
                        "source_index": {"type": "integer"},
                        "content": {
                            "type": "text",
                            "analyzer": "cjk",
                            "search_analyzer": "cjk",
                        },
                        "created_at": {"type": "date"},
                    }
                },
            },
        )

    def statistics(self) -> dict[str, object]:
        result: dict[str, object] = {
            "enabled": self.enabled,
            "available": False,
            "index": self.index_name,
            "documents": None,
            "error": self.error,
        }
        if not self.enabled or not self.available:
            result["error"] = self.error
            return result
        try:
            self.ensure_index()
            result["available"] = True
            result["documents"] = int(self.client.count(index=self.index_name)["count"])
            result["error"] = ""
        except Exception as exc:
            self.error = f"{type(exc).__name__}: {exc}"
            result["error"] = self.error
        return result

    def corpus_signature(self) -> tuple[int, int]:
        self.ensure_index()
        response = self._require_client().search(
            index=self.index_name,
            body={
                "size": 0,
                "aggs": {"max_chunk_id": {"max": {"field": "chunk_id"}}},
            },
        )
        total = response.get("hits", {}).get("total", {})
        count = int(total.get("value", 0) if isinstance(total, dict) else total or 0)
        maximum = response.get("aggregations", {}).get("max_chunk_id", {}).get("value")
        return count, int(maximum or 0)

    def search(self, query: str, top_k: int) -> list[LexicalHit]:
        if not query.strip() or top_k <= 0:
            return []
        self.ensure_index()
        response = self._require_client().search(
            index=self.index_name,
            body={
                "size": top_k,
                "_source": ["chunk_id", "source", "content"],
                "query": {"match": {"content": {"query": query, "operator": "or"}}},
            },
        )
        results = []
        for hit in response.get("hits", {}).get("hits", []):
            source = hit.get("_source", {})
            chunk_id = source.get("chunk_id")
            results.append(
                LexicalHit(
                    chunk_id=int(chunk_id) if chunk_id is not None else None,
                    source=str(source.get("source", "")),
                    content=str(source.get("content", "")),
                    score=float(hit.get("_score") or 0.0),
                )
            )
        return results

    def replace_source(self, source: str, chunks: Iterable[KnowledgeChunk]) -> int:
        self.ensure_index()
        client = self._require_client()
        client.delete_by_query(
            index=self.index_name,
            body={"query": {"term": {"source": source}}},
            conflicts="proceed",
            refresh=True,
        )
        return self._bulk_index(chunks, refresh="wait_for")

    def rebuild(self, chunks: Iterable[KnowledgeChunk]) -> int:
        client = self._require_client()
        if client.indices.exists(index=self.index_name):
            client.indices.delete(index=self.index_name)
        self.ensure_index()
        return self._bulk_index(chunks, refresh="wait_for")

    def _bulk_index(self, chunks: Iterable[KnowledgeChunk], refresh: str) -> int:
        actions = [self._action(chunk) for chunk in chunks if chunk.id is not None]
        if not actions:
            return 0
        success, _ = helpers.bulk(
            self._require_client(),
            actions,
            refresh=refresh,
            raise_on_error=True,
        )
        return int(success)

    def _action(self, chunk: KnowledgeChunk) -> dict:
        return {
            "_op_type": "index",
            "_index": self.index_name,
            "_id": str(chunk.id),
            "_source": {
                "chunk_id": int(chunk.id),
                "source": chunk.source,
                "source_index": int(chunk.source_index),
                "content": chunk.content,
                "created_at": chunk.created_at.isoformat() if chunk.created_at else None,
            },
        }

    def _create_client(self):
        raw_url = str(getattr(self.settings, "opensearch_url", "http://127.0.0.1:19200"))
        parsed = urlparse(raw_url)
        if not parsed.hostname:
            raise ValueError(f"Invalid OPENSEARCH_URL: {raw_url}")
        username = str(getattr(self.settings, "opensearch_username", ""))
        password = str(getattr(self.settings, "opensearch_password", ""))
        options = {
            "hosts": [{"host": parsed.hostname, "port": parsed.port or (443 if parsed.scheme == "https" else 80)}],
            "use_ssl": parsed.scheme == "https",
            "verify_certs": bool(getattr(self.settings, "opensearch_verify_certs", False)),
            "timeout": float(getattr(self.settings, "opensearch_timeout_seconds", 2.0)),
            "max_retries": 1,
            "retry_on_timeout": False,
            "http_compress": True,
        }
        if username:
            options["http_auth"] = (username, password)
        return OpenSearch(**options)

    def _require_client(self):
        if self.client is None:
            raise RuntimeError("OpenSearch client is unavailable")
        return self.client
