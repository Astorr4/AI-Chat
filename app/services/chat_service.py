import asyncio
import json
import time
import uuid
import hashlib

from core.database import (
    get_last_messages,
    get_relevant_messages,
    save_rag_metrics,
    save_message,
)
from config import MAX_CONTEXT_CHARS, MIN_SCORE_THRESHOLD
from config import (
    GENERATION_MAX_ACTIVE,
    GENERATION_TTL_SECONDS,
    IDEMPOTENCY_WINDOW_SECONDS,
)


class ChatService:
    def __init__(self, rag, llm_semaphore, logger):
        self.rag = rag
        self.llm_semaphore = llm_semaphore
        self.logger = logger
        self._generations = {}
        self._idempotency_index = {}
        self._session_state = {}

    def _active_generation_count(self):
        return sum(1 for state in self._generations.values() if not state.get("done"))

    def can_access_generation(self, request_id, session_id):
        state = self._generations.get(request_id)
        if not state:
            return False, "not_found"
        if state.get("session_id") != session_id:
            return False, "forbidden"
        return True, "ok"

    def _drop_generation(self, request_id):
        self._generations.pop(request_id, None)

        stale_keys = [
            key
            for key, meta in self._idempotency_index.items()
            if meta.get("request_id") == request_id
        ]
        for key in stale_keys:
            self._idempotency_index.pop(key, None)

    def _cleanup_expired_generations(self):
        now = time.time()

        expired_request_ids = []
        for request_id, state in self._generations.items():
            created_at = state.get("created_at", now)
            if now - created_at > GENERATION_TTL_SECONDS:
                expired_request_ids.append(request_id)

        for request_id in expired_request_ids:
            self._generations.pop(request_id, None)

        expired_keys = []
        for key, meta in self._idempotency_index.items():
            if now - meta.get("created_at", now) > IDEMPOTENCY_WINDOW_SECONDS:
                expired_keys.append(key)

        for key in expired_keys:
            self._idempotency_index.pop(key, None)

        stale_sessions = []
        for session_id, state in self._session_state.items():
            if now - state.get("updated_at", now) > GENERATION_TTL_SECONDS * 4:
                stale_sessions.append(session_id)

        for session_id in stale_sessions:
            self._session_state.pop(session_id, None)

    def _restore_session_state(self, session_id, previous_messages):
        state = self._session_state.get(session_id)
        if state:
            state["updated_at"] = time.time()
            return state

        last_user_questions = [
            m["content"].strip()
            for m in previous_messages
            if m["role"] == "user" and m["content"]
        ]

        state = {
            "last_intent": "fact",
            "last_focus": last_user_questions[-1] if last_user_questions else "",
            "last_mode": "precise",
            "decomposition_count": 0,
            "pending_clarification": False,
            "updated_at": time.time(),
        }
        self._session_state[session_id] = state
        return state

    def _is_ambiguous_question(self, question, previous_user_messages):
        q = (question or "").strip().lower()
        if not q:
            return False

        pronoun_markers = {
            "это", "этот", "эта", "эти", "там", "он", "она", "они", "его", "ее", "их"
        }
        tokens = q.split()

        if len(tokens) <= 2 and not previous_user_messages:
            return True

        if len(tokens) <= 3 and any(t in pronoun_markers for t in tokens) and not previous_user_messages:
            return True

        return False

    def _build_idempotency_key(self, session_id, question):
        raw = f"{session_id}|{(question or '').strip().lower()}"
        return hashlib.sha256(raw.encode("utf-8")).hexdigest()

    def _make_backpressure_error_stream(self):
        async def event_generator():
            yield "Временная перегрузка сервиса. Повторите запрос через несколько секунд."
        return event_generator()

    def _create_generation_state(self, request_id):
        self._generations[request_id] = {
            "buffer": "",
            "done": False,
            "error": None,
            "session_id": None,
            "created_at": time.time(),
        }

    def _append_generation_chunk(self, request_id, chunk):
        state = self._generations.get(request_id)
        if not state:
            return
        state["buffer"] += chunk

    def _finish_generation(self, request_id, *, error=None):
        state = self._generations.get(request_id)
        if not state:
            return
        state["done"] = True
        state["error"] = error

    def start_generation(self, *, question, session_id, user_login, custom_messages=None, is_file_analysis=False):
        self._cleanup_expired_generations()

        key = self._build_idempotency_key(session_id, question)
        existing = self._idempotency_index.get(key)
        if existing:
            request_id = existing.get("request_id")
            if request_id in self._generations:
                return request_id
            self._idempotency_index.pop(key, None)

        if self._active_generation_count() >= GENERATION_MAX_ACTIVE:
            return None

        request_id = str(uuid.uuid4())
        self._create_generation_state(request_id)
        self._generations[request_id]["session_id"] = session_id
        self._generations[request_id]["custom_messages"] = custom_messages
        self._generations[request_id]["is_file_analysis"] = is_file_analysis
        self._idempotency_index[key] = {
            "request_id": request_id,
            "created_at": time.time(),
        }
        asyncio.create_task(
            self._run_generation(
                request_id=request_id,
                question=question,
                session_id=session_id,
                user_login=user_login,
                custom_messages=custom_messages,
                is_file_analysis=is_file_analysis
            )
        )
        return request_id

    def stream_generation(self, request_id):
        async def event_generator():
            self._cleanup_expired_generations()
            state = self._generations.get(request_id)
            if not state:
                yield "###ERROR###\nnot_found"
                return

            cursor = 0
            while True:
                state = self._generations.get(request_id)
                if not state:
                    return

                buffer_text = state["buffer"]
                if len(buffer_text) > cursor:
                    chunk = buffer_text[cursor:]
                    cursor = len(buffer_text)
                    yield chunk
                    continue

                if state["done"]:
                    self._drop_generation(request_id)
                    return

                await asyncio.sleep(0.05)

        return event_generator()

    async def _run_generation(
        self,
        *,
        request_id,
        question,
        session_id,
        user_login,
        custom_messages=None,
        is_file_analysis=False
    ):
        try:
            await self._generate_answer(
                request_id=request_id,
                question=question,
                session_id=session_id,
                user_login=user_login,
                custom_messages=custom_messages,
                is_file_analysis=is_file_analysis
            )
            self._finish_generation(request_id)
        except Exception as e:
            self.logger.exception("Generation error")
            self._append_generation_chunk(
                request_id,
                "В документации информация не найдена.",
            )
            await asyncio.to_thread(
                save_message,
                session_id,
                "assistant",
                "В документации информация не найдена.",
            )
            self._finish_generation(request_id, error=str(e))

    async def _generate_answer(
        self,
        *,
        request_id,
        question,
        session_id,
        user_login,
        custom_messages=None,
        is_file_analysis=False
    ):
        retrieval_start = time.time()

        if is_file_analysis and custom_messages:
            # For file analysis, we skip the RAG pipeline and use custom messages
            messages = custom_messages
            full_answer = ""
            llm_start = time.time()

            async with self.llm_semaphore:
                llm_stream_iterator = self.rag.llm.stream(messages)

                while True:
                    token = await asyncio.to_thread(
                        lambda: next(llm_stream_iterator, None)
                    )
                    if token is None:
                        break

                    full_answer += token
                    self._append_generation_chunk(request_id, token)
                    await asyncio.sleep(0)

            llm_time = round(time.time() - llm_start, 3)
            total_time = round(time.time() - retrieval_start, 3)
            answer_length = len(full_answer)

            # Extract filename from custom messages for sources
            file_name = "Файл"
            for msg in custom_messages:
                if msg.get("role") == "user" and "Документ:" in msg.get("content", ""):
                    try:
                        file_name = msg["content"].split("Документ:")[1].split("\n")[0].strip()
                    except:
                        file_name = "Файл"
                    break

            self._append_generation_chunk(request_id, "\n###SOURCES###\n")
            self._append_generation_chunk(
                request_id,
                json.dumps([file_name]),
            )
            self._append_generation_chunk(request_id, "\n###CONFIDENCE###\n")
            self._append_generation_chunk(request_id, str(round(0.9, 3)))  # High confidence for file analysis

            await asyncio.to_thread(
                save_message,
                session_id,
                "assistant",
                full_answer,
            )

            # Save basic metrics for file analysis
            await asyncio.to_thread(
                save_rag_metrics,
                session_id=session_id,
                user_login=user_login,
                question=question,
                found_docs=1,
                filtered_docs=1,
                confidence=0.9,
                avg_score=0.9,
                min_score=0.9,
                max_score=0.9,
                coverage=1.0,
                threshold=0.0,
                sources_count=1,
                context_chars=len(custom_messages[1]["content"]) if len(custom_messages) > 1 else 0,
                retrieval_time=0,
                llm_time=llm_time,
                total_time=total_time,
                answer_length=answer_length,
                is_followup=0,
                memory_size=0,
                rejected_reason=None,
                similarity=None,
                topic_mode="file_analysis"
            )
        else:
            # Original RAG pipeline
            previous_messages = await asyncio.to_thread(
                get_relevant_messages,
                session_id,
                question,
                10,
            )

            current_question = (question or "").strip()
            if current_question and previous_messages:
                for idx in range(len(previous_messages) - 1, -1, -1):
                    msg = previous_messages[idx]
                    if (
                        msg["role"] == "user"
                        and (msg["content"] or "").strip() == current_question
                    ):
                        previous_messages.pop(idx)
                        break

            previous_user_messages = [
                m for m in previous_messages
                if m["role"] == "user"
            ]
            session_state = self._restore_session_state(session_id, previous_messages)

            is_followup = 1 if len(previous_user_messages) > 0 else 0
            memory_size = len(previous_user_messages)

            session_summary = " ".join(
                m["content"].strip()
                for m in previous_messages
                if m["role"] == "user" and m["content"]
            )
            session_summary = session_summary[:400]

            intent = self.rag.detect_intent(question)

            if self._is_ambiguous_question(question, previous_user_messages):
                clarification = (
                    "Вопрос выглядит неоднозначным. "
                    "Уточните, пожалуйста, предмет запроса: документ, процесс или конкретный показатель."
                )
                self._append_generation_chunk(request_id, clarification)
                await asyncio.to_thread(
                    save_message,
                    session_id,
                    "assistant",
                    clarification,
                )
                session_state["pending_clarification"] = True
                session_state["updated_at"] = time.time()
                return

            if intent == "followup" and not previous_user_messages:
                clarification = (
                    "Похоже, это уточняющий вопрос без контекста. "
                    "Уточните, пожалуйста, о каком процессе или документе идёт речь."
                )
                self._append_generation_chunk(request_id, clarification)
                await asyncio.to_thread(
                    save_message,
                    session_id,
                    "assistant",
                    clarification,
                )
                return

            cleaned_question = self.rag.rewrite_query(
                question,
                chat_memory=previous_messages,
                session_summary=session_summary,
                topic_mode="weak" if is_followup else "new",
            )

            subqueries = self.rag.split_into_subqueries(cleaned_question)
            if not subqueries:
                subqueries = [cleaned_question]

            dynamic_k = self.rag.dynamic_top_k(
                question,
                len(subqueries),
                topic_mode="new",
            )

            all_results = []
            similarity = None
            topic_mode = "new"

            for sq in subqueries:
                result = await asyncio.to_thread(
                    self.rag.search,
                    sq,
                    dynamic_k,
                    session_id,
                    intent,
                )

                res = result["documents"]
                similarity = result.get("similarity")
                topic_mode = result.get("topic_mode")

                all_results.extend(res)

            if not all_results:
                total_time = round(time.time() - retrieval_start, 3)

                await asyncio.to_thread(
                    self._save_rejected_metrics,
                    session_id=session_id,
                    user_login=user_login,
                    question=question,
                    found_docs=0,
                    filtered_docs=0,
                    confidence=0,
                    avg_score=0,
                    min_score=0,
                    retrieval_time=total_time,
                    is_followup=is_followup,
                    memory_size=memory_size,
                    rejected_reason="no_results",
                )

                no_result_text = "В документации информация не найдена."
                self._append_generation_chunk(request_id, no_result_text)
                await asyncio.to_thread(
                    save_message,
                    session_id,
                    "assistant",
                    no_result_text,
                )
                return

            unique_results = []
            seen = set()

            for r in all_results:
                if r["text"] not in seen:
                    unique_results.append(r)
                    seen.add(r["text"])

            found_docs = len(unique_results)

            scores = [
                r.get("hybrid_score", r.get("score", 0))
                for r in unique_results
            ]

            confidence_raw = max(scores) if scores else 0
            avg_score = sum(scores) / len(scores) if scores else 0
            min_score = min(scores) if scores else 0

            adaptive_thresh = self.rag.adaptive_threshold(
                MIN_SCORE_THRESHOLD,
                topic_mode,
                similarity,
            )

            if confidence_raw < adaptive_thresh:
                retrieval_time = round(time.time() - retrieval_start, 3)

                await asyncio.to_thread(
                    self._save_rejected_metrics,
                    session_id=session_id,
                    user_login=user_login,
                    question=question,
                    found_docs=found_docs,
                    filtered_docs=0,
                    confidence=confidence_raw,
                    avg_score=avg_score,
                    min_score=min_score,
                    retrieval_time=retrieval_time,
                    is_followup=is_followup,
                    memory_size=memory_size,
                    rejected_reason="threshold",
                )

                no_result_text = "В документации информация не найдена."
                self._append_generation_chunk(request_id, no_result_text)
                await asyncio.to_thread(
                    save_message,
                    session_id,
                    "assistant",
                    no_result_text,
                )
                return

            filtered_results = [
                r for r in unique_results
                if r.get("hybrid_score", r.get("score", 0))
                >= 0.8 * confidence_raw
            ]

            if not filtered_results:
                filtered_results = [
                    max(
                        unique_results,
                        key=lambda x: x.get("hybrid_score", x.get("score", 0)),
                    )
                ]

            filtered_docs = len(filtered_results)

            is_valid, coverage_or_reason = self.rag.retrieval_self_check(
                cleaned_question,
                filtered_results,
                topic_mode,
                similarity,
            )

            if not is_valid:
                retrieval_time = round(time.time() - retrieval_start, 3)

                await asyncio.to_thread(
                    self._save_rejected_metrics,
                    session_id=session_id,
                    user_login=user_login,
                    question=question,
                    found_docs=found_docs,
                    filtered_docs=filtered_docs,
                    confidence=0.0,
                    avg_score=avg_score,
                    min_score=min_score,
                    retrieval_time=retrieval_time,
                    is_followup=is_followup,
                    memory_size=memory_size,
                    rejected_reason=f"self_check:{coverage_or_reason}",
                )

                no_result_text = "В документации информация не найдена."
                self._append_generation_chunk(request_id, no_result_text)
                await asyncio.to_thread(
                    save_message,
                    session_id,
                    "assistant",
                    no_result_text,
                )
                return

            coverage = coverage_or_reason

            context = ""
            sources = []
            current_length = 0

            for r in filtered_results:
                chunk = r["text"]
                doc_type = r.get("doc_type")

                if current_length + len(chunk) > MAX_CONTEXT_CHARS:
                    break

                if doc_type == "excel_row":
                    formatted = f"[EXCEL ROW]\n{chunk}\n"
                else:
                    formatted = f"[TEXT CHUNK]\n{chunk}\n"

                context += formatted + "\n"
                current_length += len(formatted)
                sources.append(r["source"])

            context_chars = len(context)
            sources_count = len(set(sources))
            retrieval_time = round(time.time() - retrieval_start, 3)

            final_confidence = self.rag.recalibrate_confidence(
                max_score=confidence_raw,
                avg_score=avg_score,
                coverage=coverage,
                similarity=similarity,
                topic_mode=topic_mode,
            )

            messages = self._build_grounded_messages(
                context=context,
                question=question,
            )

            if final_confidence < 0.35:
                clarification = self._build_clarification_question(question)
                self._append_generation_chunk(request_id, clarification)
                await asyncio.to_thread(
                    save_message,
                    session_id,
                    "assistant",
                    clarification,
                )
                await asyncio.to_thread(
                    self._save_rejected_metrics,
                    session_id=session_id,
                    user_login=user_login,
                    question=question,
                    found_docs=found_docs,
                    filtered_docs=filtered_docs,
                    confidence=final_confidence,
                    avg_score=avg_score,
                    min_score=min_score,
                    retrieval_time=retrieval_time,
                    is_followup=is_followup,
                    memory_size=memory_size,
                    rejected_reason="low_confidence_clarify",
                )
                return

            full_answer = ""
            llm_start = time.time()

            async with self.llm_semaphore:
                llm_stream_iterator = self.rag.llm.stream(messages)

                while True:
                    token = await asyncio.to_thread(
                        lambda: next(llm_stream_iterator, None)
                    )
                    if token is None:
                        break

                    full_answer += token
                    self._append_generation_chunk(request_id, token)
                    await asyncio.sleep(0)

            session_state["last_intent"] = intent
            session_state["last_focus"] = cleaned_question[:300]
            session_state["last_mode"] = "standard"
            session_state["decomposition_count"] = session_state.get("decomposition_count", 0) + (1 if len(subqueries) > 1 else 0)
            session_state["pending_clarification"] = False
            session_state["updated_at"] = time.time()

            llm_time = round(time.time() - llm_start, 3)
            total_time = round(time.time() - retrieval_start, 3)
            answer_length = len(full_answer)

            self._append_generation_chunk(request_id, "\n###SOURCES###\n")
            self._append_generation_chunk(
                request_id,
                json.dumps(list(set(sources))),
            )
            self._append_generation_chunk(request_id, "\n###CONFIDENCE###\n")
            self._append_generation_chunk(request_id, str(round(final_confidence, 3)))

            await asyncio.to_thread(
                save_message,
                session_id,
                "assistant",
                full_answer,
            )

            metrics_payload = dict(
                session_id=session_id,
                question=question,
                found_docs=found_docs,
                filtered_docs=filtered_docs,
                confidence=final_confidence,
                avg_score=avg_score,
                min_score=min_score,
                max_score=confidence_raw,
                coverage=coverage,
                threshold=adaptive_thresh,
                sources_count=sources_count,
                context_chars=context_chars,
                retrieval_time=retrieval_time,
                llm_time=llm_time,
                total_time=total_time,
                answer_length=answer_length,
                is_followup=is_followup,
                memory_size=memory_size,
                rejected_reason=None,
                topic_mode=intent,
            )
            if final_confidence < 0.8:
                asyncio.create_task(
                    self._background_verification_and_metrics(
                        session_id,
                        user_login,
                        question,
                        context,
                        full_answer,
                        final_confidence,
                        metrics_payload,
                    )
                )
            else:
                metrics_payload["user_login"] = user_login
                await asyncio.to_thread(save_rag_metrics, **metrics_payload)

    def _save_rejected_metrics(
        self,
        *,
        session_id,
        user_login,
        question,
        found_docs,
        filtered_docs,
        confidence,
        avg_score,
        min_score,
        retrieval_time,
        is_followup,
        memory_size,
        rejected_reason,
    ):
        save_rag_metrics(
            session_id=session_id,
            user_login=user_login,
            question=question,
            found_docs=found_docs,
            filtered_docs=filtered_docs,
            confidence=confidence,
            avg_score=avg_score,
            min_score=min_score,
            sources_count=0,
            context_chars=0,
            retrieval_time=retrieval_time,
            llm_time=0,
            total_time=retrieval_time,
            answer_length=0,
            is_followup=is_followup,
            memory_size=memory_size,
            rejected_reason=rejected_reason,
        )

    def _build_grounded_messages(self, *, context, question):
        return [
            {
                "role": "system",
                "content": (
                    "Ты корпоративный AI-помощник. "
                    "Отвечай только на основе предоставленного контекста. "
                    "Если информации недостаточно, сообщи об этом и задай 1 уточняющий вопрос. "
                    "Дай точный структурированный ответ без лишних деталей."
                ),
            },
            {
                "role": "user",
                "content": f"Контекст:\n{context}\n\nВопрос:\n{question}",
            },
        ]

    def _build_clarification_question(self, question):
        return (
            "Не хватает уверенности для точного ответа. "
            "Уточните, пожалуйста: "
            f"что именно по теме «{question[:120]}» нужно — определение, шаги или сравнение?"
        )

    async def _background_verification_and_metrics(
        self,
        session_id,
        user_login,
        question,
        context,
        answer,
        final_confidence,
        metrics_payload,
    ):
        try:
            verification = await asyncio.to_thread(
                self.rag.verify_answer,
                question,
                context,
                answer,
            )

            if verification == "PARTIAL":
                final_confidence *= 0.8
            elif verification == "INVALID":
                final_confidence = 0.0

            metrics_payload["confidence"] = final_confidence
            metrics_payload["user_login"] = user_login
            metrics_payload["rejected_reason"] = (
                "verification_invalid"
                if verification == "INVALID"
                else None
            )

            await asyncio.to_thread(save_rag_metrics, **metrics_payload)

        except Exception:
            self.logger.exception("Background verification error")

    def stream_chat(self, *, question, session_id, user_login, custom_messages=None, is_file_analysis=False):
        request_id = self.start_generation(
            question=question,
            session_id=session_id,
            user_login=user_login,
            custom_messages=custom_messages,
            is_file_analysis=is_file_analysis
        )

        if request_id is None:
            return None, self._make_backpressure_error_stream()

        return request_id, self.stream_generation(request_id)
