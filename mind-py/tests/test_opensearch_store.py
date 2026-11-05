from __future__ import annotations

import unittest
from types import SimpleNamespace
from unittest.mock import patch

from app.services.search_store import OpenSearchKnowledgeStore


class _Indices:
    def __init__(self):
        self.created = None

    def exists(self, index):
        return self.created is not None

    def create(self, index, body):
        self.created = (index, body)


class _Client:
    def __init__(self):
        self.indices = _Indices()
        self.search_response = {"hits": {"hits": []}}

    def ping(self):
        return True

    def search(self, **kwargs):
        return self.search_response


def _settings():
    return SimpleNamespace(
        knowledge_search_enabled=True,
        knowledge_search_required=False,
        opensearch_url="http://127.0.0.1:19200",
        opensearch_index="test_chunks",
        opensearch_username="",
        opensearch_password="",
        opensearch_verify_certs=False,
        opensearch_timeout_seconds=1,
    )


class OpenSearchKnowledgeStoreTests(unittest.TestCase):
    def test_index_uses_persistent_cjk_bm25_mapping(self):
        client = _Client()
        with patch("app.services.search_store.OpenSearch", return_value=client):
            store = OpenSearchKnowledgeStore(_settings())
            store.ensure_index()

        index, body = client.indices.created
        self.assertEqual(index, "test_chunks")
        self.assertEqual(body["mappings"]["properties"]["content"]["analyzer"], "cjk")
        self.assertEqual(body["settings"]["number_of_replicas"], 0)

    def test_search_converts_opensearch_hits(self):
        client = _Client()
        client.indices.created = ("test_chunks", {})
        client.search_response = {
            "hits": {
                "hits": [
                    {
                        "_score": 3.5,
                        "_source": {
                            "chunk_id": 8,
                            "source": "sleep.md",
                            "content": "固定作息有助于改善睡眠",
                        },
                    }
                ]
            }
        }
        with patch("app.services.search_store.OpenSearch", return_value=client):
            store = OpenSearchKnowledgeStore(_settings())
            hits = store.search("失眠", 16)

        self.assertEqual(len(hits), 1)
        self.assertEqual(hits[0].chunk_id, 8)
        self.assertEqual(hits[0].source, "sleep.md")
        self.assertEqual(hits[0].score, 3.5)


if __name__ == "__main__":
    unittest.main()
