import base64
import unittest
from datetime import datetime, timedelta

from fastapi import FastAPI
from fastapi.testclient import TestClient
from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker
from sqlalchemy.pool import StaticPool

from app.api.routes import router
from app.core.database import Base
from app.core.database import get_db
from app.core.security import hash_password
from app.models.entities import ChatMessage, ChatSession, UserAccount
from app.services.report import ReportService


class StudentChatHistoryTests(unittest.TestCase):
    def setUp(self):
        self.engine = create_engine(
            "sqlite:///:memory:",
            connect_args={"check_same_thread": False},
            poolclass=StaticPool,
        )
        Base.metadata.create_all(self.engine)
        self.db = sessionmaker(bind=self.engine)()

        self.student = UserAccount(
            username="student-a",
            display_name="Student A",
            password_hash=hash_password("password-a"),
            roles_csv="ROLE_USER",
        )
        self.other_student = UserAccount(
            username="student-b",
            display_name="Student B",
            password_hash=hash_password("password-b"),
            roles_csv="ROLE_USER",
        )
        self.db.add_all([self.student, self.other_student])
        self.db.commit()

        earlier = datetime.utcnow() - timedelta(days=1)
        self.old_session = ChatSession(
            public_id="student-a-old",
            title="昨天的压力",
            user_id=self.student.id,
            created_at=earlier,
            updated_at=earlier,
        )
        self.new_session = ChatSession(
            public_id="student-a-new",
            title="今天的睡眠",
            user_id=self.student.id,
        )
        self.other_session = ChatSession(
            public_id="student-b-private",
            title="别人的私密对话",
            user_id=self.other_student.id,
        )
        self.db.add_all([self.old_session, self.new_session, self.other_session])
        self.db.commit()

        self.db.add_all(
            [
                ChatMessage(user_id=self.student.id, session_id=self.old_session.id, role="USER", content="昨天压力很大"),
                ChatMessage(user_id=self.student.id, session_id=self.old_session.id, role="ASSISTANT", content="我们慢慢梳理"),
                ChatMessage(user_id=self.student.id, session_id=self.new_session.id, role="USER", content="今天没睡好"),
                ChatMessage(user_id=self.other_student.id, session_id=self.other_session.id, role="USER", content="private"),
            ]
        )
        self.db.commit()
        self.service = ReportService(self.db)

        app = FastAPI()
        app.include_router(router)
        app.dependency_overrides[get_db] = lambda: self.db
        self.client = TestClient(app)

    def tearDown(self):
        self.client.close()
        self.db.close()
        self.engine.dispose()

    @staticmethod
    def auth(username: str, password: str) -> dict[str, str]:
        token = base64.b64encode(f"{username}:{password}".encode()).decode()
        return {"Authorization": f"Basic {token}"}

    def test_lists_only_current_students_sessions_newest_first(self):
        sessions = self.service.student_sessions(self.student.id)

        self.assertEqual([item.sessionId for item in sessions], ["student-a-new", "student-a-old"])
        self.assertEqual([item.messageCount for item in sessions], [1, 2])
        self.assertNotIn("student-b-private", [item.sessionId for item in sessions])

    def test_loads_owned_conversation_in_message_order(self):
        conversation = self.service.student_conversation("student-a-old", self.student.id)

        self.assertEqual(conversation.title, "昨天的压力")
        self.assertEqual([message.role for message in conversation.messages], ["USER", "ASSISTANT"])
        self.assertEqual(conversation.messages[0].content, "昨天压力很大")

    def test_cannot_load_another_students_conversation(self):
        with self.assertRaisesRegex(ValueError, "Session not found"):
            self.service.student_conversation("student-b-private", self.student.id)

    def test_student_history_endpoints_enforce_ownership(self):
        headers = self.auth("student-a", "password-a")

        sessions = self.client.get("/api/chat/sessions", headers=headers)
        owned = self.client.get("/api/chat/sessions/student-a-old", headers=headers)
        private = self.client.get("/api/chat/sessions/student-b-private", headers=headers)

        self.assertEqual(sessions.status_code, 200)
        self.assertEqual([item["sessionId"] for item in sessions.json()], ["student-a-new", "student-a-old"])
        self.assertEqual(owned.status_code, 200)
        self.assertEqual(len(owned.json()["messages"]), 2)
        self.assertEqual(private.status_code, 404)


if __name__ == "__main__":
    unittest.main()
