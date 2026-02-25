import asyncio
import json
import time

from core.database import get_last_messages, save_rag_metrics
from config import MAX_CONTEXT_CHARS, MIN_SCORE_THRESHOLD


class ChatService:
    def __init__(self, rag, llm_semaphore, logger):
        self.rag = rag
        self.llm_semaphore = llm_semaphore
        self.logger = logger

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

    def stream_chat(self, *, question, session_id, user_login):
        async def event_generator():
            retrieval_start = time.time()

            previous_messages = await asyncio.to_thread(
                get_last_messages,
                session_id,
                10,
            )
            previous_messages = previous_messages[:-1] if previous_messages else []

            previous_user_messages = [
                m for m in previous_messages
                if m["role"] == "user"
            ]

            is_followup = 1 if len(previous_user_messages) > 0 else 0
            memory_size = len(previous_user_messages)

            cleaned_question = self.rag.normalize_query(question)

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

                yield "В документации информация не найдена."
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

                yield "В документации информация не найдена."
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

                yield "В документации информация не найдена."
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

            messages = [
                {
                    "role": "system",
                    "content": """
Ты корпоративный AI-помощник.
Отвечай строго только на основе предоставленного контекста.
Не добавляй информацию вне контекста.
""",
                },
                {
                    "role": "user",
                    "content": f"""
Контекст:
{context}

Вопрос:
{question}
""",
                },
            ]

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
                    yield token
                    await asyncio.sleep(0)

                yield "\n###SOURCES###\n"
                yield json.dumps(list(set(sources)))

                yield "\n###CONFIDENCE###\n"
                yield str(round(final_confidence, 3))

            llm_time = round(time.time() - llm_start, 3)
            total_time = round(time.time() - retrieval_start, 3)
            answer_length = len(full_answer)

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

        return event_generator()
