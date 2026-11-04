from __future__ import annotations

import json
import logging

from sqlalchemy.orm import Session

from app.agents.harness import MindBridgeAgentHarness
from app.agents.autonomous import STATIC_SAFETY_RESPONSE
from app.core.config import Settings
from app.models.entities import UserAccount
from app.schemas.dtos import ChatRequest, ChatStreamEvent
from app.services.ai import AiClient


logger = logging.getLogger(__name__)


class ChatService:
    def __init__(self, db: Session, settings: Settings):
        self.db = db
        self.settings = settings
        self.ai = AiClient(settings)
        self.agent_harness = MindBridgeAgentHarness(db, settings)

    async def stream_chat(self, user: UserAccount, request: ChatRequest):
        outcome = self.agent_harness.run(user, request)
        yield sse("meta", ChatStreamEvent(type="meta", sessionId=outcome.session.public_id).model_dump(by_alias=True))
        tools_dispatched = False
        if outcome.workflow_status in {"FALLBACK_COMPLETED", "FALLBACK_REJECTED"}:
            try:
                await self.agent_harness.dispatch_tools(outcome.tool_plan)
                tools_dispatched = True
            except Exception as exc:
                logger.warning(
                    "Immediate fallback alert dispatch failed for session=%s report_id=%s: %s",
                    outcome.session.public_id,
                    outcome.report_id,
                    exc,
                    exc_info=True,
                )
        assistant = []
        if outcome.direct_response_text:
            chunks = [outcome.direct_response_text[index:index + 12] for index in range(0, len(outcome.direct_response_text), 12)]
            for token in chunks:
                assistant.append(token)
                yield sse("token", ChatStreamEvent(type="token", sessionId=outcome.session.public_id, content=token).model_dump())
        else:
            attempts = max(1, int(getattr(self.settings, "ai_final_retry_attempts", 1)) + 1)
            model_error = None
            for attempt in range(attempts):
                emitted = False
                try:
                    async for token in self.ai.stream(outcome.response_messages):
                        if not token:
                            continue
                        emitted = True
                        assistant.append(token)
                        yield sse("token", ChatStreamEvent(type="token", sessionId=outcome.session.public_id, content=token).model_dump())
                    if emitted:
                        model_error = None
                        break
                    model_error = RuntimeError("final model returned empty content")
                except Exception as exc:
                    model_error = exc
                    if emitted:
                        break
                if attempt + 1 < attempts:
                    logger.warning("Final model call failed before output; retrying once: %s", model_error)
            if not assistant and model_error is not None:
                fallback = STATIC_SAFETY_RESPONSE if outcome.workflow_status == "FALLBACK_COMPLETED" else "服务暂时无法生成回复，请稍后再试。"
                assistant.append(fallback)
                yield sse("token", ChatStreamEvent(type="token", sessionId=outcome.session.public_id, content=fallback).model_dump())
                self.agent_harness.record_delivery_status(outcome.trace_id, "MODEL_FAILED", str(model_error))
        if assistant:
            self.agent_harness.save_assistant_message(user, outcome.session, "".join(assistant))
        if not tools_dispatched:
            try:
                await self.agent_harness.dispatch_tools(outcome.tool_plan)
            except Exception as exc:
                logger.warning(
                    "Post-response tool dispatch failed for session=%s report_id=%s: %s",
                    outcome.session.public_id,
                    outcome.report_id,
                    exc,
                    exc_info=True,
                )
        yield sse("done", ChatStreamEvent(type="done", sessionId=outcome.session.public_id).model_dump())


def sse(event: str, data: dict) -> str:
    return f"event: {event}\ndata: {json.dumps(data, ensure_ascii=False, default=str)}\n\n"
