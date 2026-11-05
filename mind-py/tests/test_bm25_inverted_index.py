from __future__ import annotations

import unittest
from types import SimpleNamespace
from unittest.mock import patch

from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker

from app.models.entities import KnowledgeChunk
from app.services import knowledge as knowledge_module
from app.services.knowledge import (
    Bm25InvertedIndex,
    KnowledgeCorpusSignature,
    KnowledgeService,
    bm25_scores,
)


class _UnavailableVectorStore:
    can_embed = False
    error = "disabled for unit test"

    def __init__(self, settings):
        self.settings = settings


def _chunk(chunk_id: int, source: str, content: str) -> KnowledgeChunk:
    return KnowledgeChunk(id=chunk_id, source=source, source_index=0, content=content)


class Bm25InvertedIndexTests(unittest.TestCase):
    def test_cached_index_matches_one_shot_bm25_scores(self):
        chunks = [
            _chunk(1, "exam.md", "考试焦虑会影响睡眠，可以通过呼吸练习缓解焦虑。"),
            _chunk(2, "sleep.md", "失眠时保持固定作息并减少睡前刺激。"),
            _chunk(3, "code.md", "Python exception handling uses try and except."),
            _chunk(4, "support.md", "考试压力较大时可以联系学校心理中心。"),
        ]
        query = "考试焦虑失眠"
        expected = bm25_scores(query, chunks)
        index = Bm25InvertedIndex()
        index.rebuild(chunks, KnowledgeCorpusSignature(len(chunks), 4, None))

        actual = dict(index.search(query, top_k=16))

        self.assertEqual(set(actual), set(expected))
        for chunk_id, score in expected.items():
            self.assertAlmostEqual(actual[chunk_id], score, places=12)

    def test_replacing_source_removes_old_postings_and_adds_new_ones(self):
        index = Bm25InvertedIndex()
        index.rebuild(
            [_chunk(1, "guide.md", "旧版内容只讨论考试"), _chunk(2, "other.md", "睡眠建议")],
            KnowledgeCorpusSignature(2, 2, None),
        )

        index.replace_source(
            "guide.md",
            [_chunk(3, "guide.md", "新版内容讨论惊恐发作和呼吸练习")],
            KnowledgeCorpusSignature(2, 3, None),
        )

        self.assertEqual(index.search("旧", 16), [])
        self.assertEqual(index.search("惊恐发作", 16)[0][0], 3)
        self.assertEqual(index.statistics()["documents"], 2)


class KnowledgeServiceBm25CacheTests(unittest.TestCase):
    def setUp(self):
        self.engine = create_engine("sqlite:///:memory:")
        KnowledgeChunk.__table__.create(self.engine)
        self.db = sessionmaker(bind=self.engine)()
        self.settings = SimpleNamespace(
            knowledge_chunk_size=512,
            knowledge_chunk_overlap=64,
            knowledge_top_k=4,
            knowledge_candidate_k=16,
            knowledge_hybrid_vector_weight=0.65,
            knowledge_hybrid_bm25_weight=0.35,
            knowledge_rerank_enabled=True,
        )
        self.db.add_all(
            [
                KnowledgeChunk(source="exam.md", source_index=0, content="考试焦虑和失眠支持"),
                KnowledgeChunk(source="sleep.md", source_index=0, content="固定作息改善睡眠"),
                KnowledgeChunk(source="code.md", source_index=0, content="Python代码示例"),
            ]
        )
        self.db.commit()

    def tearDown(self):
        self.db.close()
        self.engine.dispose()

    def _service(self) -> KnowledgeService:
        with patch.object(knowledge_module, "ChromaKnowledgeStore", _UnavailableVectorStore):
            return KnowledgeService(self.db, self.settings)

    def test_warm_query_tokenizes_only_query_not_every_chunk(self):
        service = self._service()
        statistics = service.warm_bm25_index()
        self.assertEqual(statistics["documents"], 3)

        original_tokenize = knowledge_module.tokenize
        with patch.object(knowledge_module, "tokenize", wraps=original_tokenize) as tracked:
            results = service._retrieve_bm25("考试焦虑", 16)

        self.assertTrue(results)
        self.assertEqual(tracked.call_count, 1)

    def test_database_signature_rebuilds_index_after_external_insert(self):
        service = self._service()
        service.warm_bm25_index()
        self.db.add(KnowledgeChunk(source="panic.md", source_index=0, content="惊恐发作时进行呼吸练习"))
        self.db.commit()

        results = service._retrieve_bm25("惊恐发作", 16)

        self.assertTrue(results)
        self.assertEqual(results[0].source, "panic.md")
        self.assertEqual(service.bm25_index.statistics()["documents"], 4)

    def test_ingest_updates_ready_index_incrementally(self):
        service = self._service()
        service.warm_bm25_index()

        service.ingest("exam.md", "新版考试支持包含渐进式放松训练")

        self.assertEqual(service._retrieve_bm25("焦虑", 16), [])
        results = service._retrieve_bm25("渐进式放松", 16)
        self.assertTrue(results)
        self.assertEqual(results[0].source, "exam.md")


if __name__ == "__main__":
    unittest.main()
