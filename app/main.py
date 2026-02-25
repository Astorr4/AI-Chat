import uuid
import asyncio
import json
import time
import logging
import re
import numpy as np
from collections import defaultdict, deque
import getpass

from fastapi import FastAPI, Request
from fastapi.staticfiles import StaticFiles
from fastapi.responses import FileResponse, StreamingResponse, JSONResponse
from pydantic import BaseModel
import sqlite3
from core.database import DB_PATH
from core.embeddings import EmbeddingModel
from core.vector_store import VectorStore
from core.llm import LLM
from core.rag import RAG
from core.database import (
    init_db,
    save_session,
    save_message,
    increment_question_stat,
    cleanup_old_data,
    get_last_messages,
    save_rag_metrics,
    get_connection
)
from config import (
    MAX_CONTEXT_CHARS,
    RATE_LIMIT,
    RATE_WINDOW,
    LLM_CONCURRENCY_LIMIT,
    MIN_SCORE_THRESHOLD,
)

# =====================================
# CONFIG
# =====================================

llm_semaphore = asyncio.Semaphore(LLM_CONCURRENCY_LIMIT)


# =====================================
# LOGGING
# =====================================

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(message)s",
)

logger = logging.getLogger("AI-Assistant")

# =====================================
# APP INIT
# =====================================

app = FastAPI(title="Corporate AI Assistant")
app.mount("/static", StaticFiles(directory="app/static"), name="static")
init_db()

# =====================================
# RAG INIT
# =====================================

embed = EmbeddingModel()
store = VectorStore()
llm = LLM()
rag = RAG(embed, store, llm)

# =====================================
# RATE LIMIT
# =====================================

rate_limit_store = defaultdict(deque)

def check_rate_limit(session_id):
    now = time.time()
    window_start = now - RATE_WINDOW

    timestamps = rate_limit_store[session_id]

    while timestamps and timestamps[0] < window_start:
        timestamps.popleft()

    if len(timestamps) >= RATE_LIMIT:
        return False

    timestamps.append(now)
    return True

async def background_verification_and_metrics(
    rag,
    session_id,
    user_login,
    question,
    context,
    answer,
    final_confidence,
    metrics_payload
):
    try:
        verification = rag.verify_answer(
            question,
            context,
            answer
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

        save_rag_metrics(**metrics_payload)

    except Exception as e:
        print("Background verification error:", e)
# =====================================
# CLEANUP TASK
# =====================================

async def cleanup_task():
    while True:
        cleanup_old_data(
            days_messages=7,
            days_metrics=30,
            days_question_stats=180
        )
        await asyncio.sleep(600)

@app.on_event("startup")
async def startup_event():
    asyncio.create_task(cleanup_task())


# =====================================
# MODELS
# =====================================

class Query(BaseModel):
    question: str


# =====================================
# ROUTES
# =====================================

@app.get("/")
def serve_ui():
    return FileResponse("app/static/index.html")


@app.post("/chat")
async def chat(query: Query, request: Request):
    user_login = getpass.getuser()
    session_id = request.headers.get("X-Session-Id")
    if not session_id:
        session_id = str(uuid.uuid4())

    if not check_rate_limit(session_id):
        return JSONResponse(
            status_code=429,
            content={"error": "Превышен лимит запросов."},
        )

    save_session(session_id)
    save_message(session_id, "user", query.question)
    increment_question_stat(query.question)

    async def event_generator():

        retrieval_start = time.time()

        previous_messages = get_last_messages(session_id, limit=4)

        # Убираем текущее сообщение
        previous_messages = get_last_messages(session_id, limit=10)

        # убираем текущее сообщение
        previous_messages = previous_messages[:-1] if previous_messages else []

        previous_user_messages = [
            m for m in previous_messages
            if m["role"] == "user"
        ]

        is_followup = 1 if len(previous_user_messages) > 0 else 0
        memory_size = len(previous_user_messages)
        # ===============================
        # NORMALIZE QUERY
        # ===============================

        cleaned_question = rag.normalize_query(query.question)

        # ===============================
        # SPLIT INTO SUBQUERIES
        # ===============================

        subqueries = rag.split_into_subqueries(cleaned_question)
        if not subqueries:
            subqueries = [cleaned_question]

        dynamic_k = rag.dynamic_top_k(
            query.question,
            len(subqueries),
            topic_mode="new"
        )

        all_results = []
        similarity = None
        topic_mode = "new"

        for sq in subqueries:

            result = rag.search(sq, top_k=dynamic_k)

            res = result["documents"]
            similarity = result.get("similarity")
            topic_mode = result.get("topic_mode")

            all_results.extend(res)

        if not all_results:

            total_time = round(time.time() - retrieval_start, 3)

            save_rag_metrics(
                session_id=session_id,
                user_login=user_login,
                question=query.question,
                found_docs=0,
                filtered_docs=0,
                confidence=0,
                avg_score=0,
                min_score=0,
                sources_count=0,
                context_chars=0,
                retrieval_time=total_time,
                llm_time=0,
                total_time=total_time,
                answer_length=0,
                is_followup=is_followup,
                memory_size=memory_size,
                rejected_reason="no_results"
            )

            yield "В документации информация не найдена."
            return

        # ===============================
        # REMOVE DUPLICATES
        # ===============================

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

        # ===============================
        # ADAPTIVE THRESHOLD
        # ===============================

        adaptive_thresh = rag.adaptive_threshold(
            MIN_SCORE_THRESHOLD,
            topic_mode,
            similarity
        )

        if confidence_raw < adaptive_thresh:

            retrieval_time = round(time.time() - retrieval_start, 3)

            save_rag_metrics(
                session_id=session_id,
                user_login=user_login,
                question=query.question,
                found_docs=found_docs,
                filtered_docs=0,
                confidence=confidence_raw,
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
                rejected_reason="threshold"
            )

            yield "В документации информация не найдена."
            return

        # ===============================
        # FILTER CONTEXT
        # ===============================

        filtered_results = [
            r for r in unique_results
            if r.get("hybrid_score", r.get("score", 0))
            >= 0.8 * confidence_raw
        ]

        if not filtered_results:
            filtered_results = [
                max(unique_results,
                    key=lambda x: x.get(
                        "hybrid_score",
                        x.get("score", 0)
                    )
                )
            ]

        filtered_docs = len(filtered_results)

        # ===============================
        # RETRIEVAL SELF-CHECK
        # ===============================

        is_valid, coverage_or_reason = rag.retrieval_self_check(
            cleaned_question,
            filtered_results,
            topic_mode,
            similarity
        )

        if not is_valid:

            retrieval_time = round(time.time() - retrieval_start, 3)

            save_rag_metrics(
                session_id=session_id,
                user_login=user_login,
                question=query.question,
                found_docs=found_docs,
                filtered_docs=filtered_docs,
                confidence=0.0,
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
                rejected_reason=f"self_check:{coverage_or_reason}"
            )

            yield "В документации информация не найдена."
            return

        coverage = coverage_or_reason

        # ===============================
        # BUILD CONTEXT
        # ===============================

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

        # ===============================
        # RECALIBRATE CONFIDENCE
        # ===============================

        final_confidence = rag.recalibrate_confidence(
            max_score=confidence_raw,
            avg_score=avg_score,
            coverage=coverage,
            similarity=similarity,
            topic_mode=topic_mode
        )

        # ===============================
        # LLM GENERATION
        # ===============================

        messages = [
            {
                "role": "system",
                "content": """
Ты корпоративный AI-помощник.
Отвечай строго только на основе предоставленного контекста.
Не добавляй информацию вне контекста.
"""
            },
            {
                "role": "user",
                "content": f"""
Контекст:
{context}

Вопрос:
{query.question}
"""
            }
        ]

        full_answer = ""
        llm_start = time.time()

        async with llm_semaphore:
            for token in rag.llm.stream(messages):
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
            question=query.question,
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
            rejected_reason=None
        )

        # ===============================
        # CONDITIONAL VERIFICATION
        # ===============================

        if final_confidence < 0.8:
            asyncio.create_task(
                background_verification_and_metrics(
                    rag,
                    session_id,
                    user_login,
                    query.question,
                    context,
                    full_answer,
                    final_confidence,
                    metrics_payload
                )
            )
        else:
            metrics_payload["user_login"] = user_login
            save_rag_metrics(**metrics_payload)

        return

    response = StreamingResponse(
        event_generator(),
        media_type="text/plain"
    )

    response.headers["X-Session-Id"] = session_id
    return response
@app.post("/chat-sync")
async def chat_sync(query: Query, request: Request):

    session_id = request.headers.get("X-Session-Id")
    if not session_id:
        session_id = str(uuid.uuid4())

    save_session(session_id)
    save_message(session_id, "user", query.question)

    retrieval_start = time.time()

    cleaned_question = rag.normalize_query(query.question)

    result = rag.search(cleaned_question, top_k=3)

    documents = result["documents"]
    similarity = result.get("similarity")
    topic_mode = result.get("topic_mode")

    if not documents:
        return {
            "answer": "В документации информация не найдена.",
            "confidence": 0,
            "rejected": True
        }

    scores = [
        d.get("hybrid_score", d.get("score", 0))
        for d in documents
    ]

    max_score = max(scores)
    avg_score = sum(scores) / len(scores)

    context = "\n".join(d["text"] for d in documents)

    messages = [
        {
            "role": "system",
            "content": "Отвечай строго по контексту."
        },
        {
            "role": "user",
            "content": f"Контекст:\n{context}\n\nВопрос:\n{query.question}"
        }
    ]

    answer = rag.llm.generate(messages)

    final_confidence = rag.recalibrate_confidence(
        max_score=max_score,
        avg_score=avg_score,
        coverage=1.0,
        similarity=similarity,
        topic_mode=topic_mode
    )
    save_rag_metrics(
        session_id=session_id,
        user_login="load_test",
        question=query.question,
        found_docs=len(documents),
        filtered_docs=len(documents),
        confidence=final_confidence,
        avg_score=avg_score,
        min_score=min(scores),
        max_score=max_score,
        coverage=1.0,
        threshold=0.0,
        sources_count=len(documents),
        context_chars=len(context),
        retrieval_time=round(time.time() - retrieval_start, 3),
        llm_time=0,
        total_time=round(time.time() - retrieval_start, 3),
        answer_length=len(answer),
        is_followup=0,
        memory_size=0,
        rejected_reason=None,
        similarity=similarity,
        topic_mode=topic_mode
    )
    return {
        "answer": answer,
        "confidence": final_confidence,
        "rejected": False
    }
@app.get("/rag-stats")
def rag_stats(from_date: str = None, to_date: str = None):

    conn = sqlite3.connect(DB_PATH)
    cursor = conn.cursor()

    conditions = []
    params = []

    if from_date and to_date:
        conditions.append("date(created_at) BETWEEN ? AND ?")
        params.extend([from_date, to_date])

    where_clause = ""
    if conditions:
        where_clause = "WHERE " + " AND ".join(conditions)

    # -------------------------
    # BASIC AGGREGATES
    # -------------------------

    cursor.execute(f"SELECT COUNT(*) FROM rag_metrics {where_clause}", params)
    total_requests = cursor.fetchone()[0]

    cursor.execute(
        f"""
        SELECT COUNT(*) FROM rag_metrics
        {where_clause}
        {"AND" if where_clause else "WHERE"} rejected_reason IS NULL
        """,
        params
    )
    success_count = cursor.fetchone()[0]

    cursor.execute(
        f"""
        SELECT rejected_reason, COUNT(*)
        FROM rag_metrics
        {where_clause}
        {"AND" if where_clause else "WHERE"} rejected_reason IS NOT NULL
        GROUP BY rejected_reason
        """,
        params
    )
    rejection_types = dict(cursor.fetchall())

    rejection_count = sum(
        v for k, v in rejection_types.items() if k is not None
    )

    rejection_rate = (
        rejection_count / total_requests
        if total_requests else 0
    )

    cursor.execute(
        f"SELECT AVG(confidence) FROM rag_metrics {where_clause}",
        params
    )
    avg_confidence = cursor.fetchone()[0] or 0

    cursor.execute(
        f"SELECT AVG(total_time) FROM rag_metrics {where_clause}",
        params
    )
    avg_total_time = cursor.fetchone()[0] or 0

    cursor.execute(
        f"SELECT AVG(llm_time) FROM rag_metrics {where_clause}",
        params
    )
    avg_llm_time = cursor.fetchone()[0] or 0

    cursor.execute(
        f"SELECT AVG(context_chars) FROM rag_metrics {where_clause}",
        params
    )
    avg_context = cursor.fetchone()[0] or 0

    # -------------------------
    # SCATTER DATA
    # -------------------------

    scatter_conditions = conditions.copy()
    scatter_conditions.append("rejected_reason IS NULL")

    scatter_where = ""
    if scatter_conditions:
        scatter_where = "WHERE " + " AND ".join(scatter_conditions)

    cursor.execute(
        f"""
        SELECT context_chars, llm_time
        FROM rag_metrics
        {scatter_where}
        """,
        params
    )

    scatter_data = [
        {"x": row[0], "y": row[1]}
        for row in cursor.fetchall()
        if row[0] and row[1]
    ]
    # -------------------------
    # CONFIDENCE DISTRIBUTION
    # -------------------------

    confidence_buckets = {
        "0-0.4": 0,
        "0.4-0.5": 0,
        "0.5-0.6": 0,
        "0.6-0.7": 0,
        "0.7-0.8": 0,
        "0.8-1.0": 0,
    }

    cursor.execute(
        f"SELECT confidence FROM rag_metrics {where_clause}",
        params
    )

    for (conf,) in cursor.fetchall():
        if conf is None:
            continue
        if conf < 0.4:
            confidence_buckets["0-0.4"] += 1
        elif conf < 0.5:
            confidence_buckets["0.4-0.5"] += 1
        elif conf < 0.6:
            confidence_buckets["0.5-0.6"] += 1
        elif conf < 0.7:
            confidence_buckets["0.6-0.7"] += 1
        elif conf < 0.8:
            confidence_buckets["0.7-0.8"] += 1
        else:
            confidence_buckets["0.8-1.0"] += 1


    # -------------------------
    # LATENCY DISTRIBUTION
    # -------------------------

    latency_buckets = {
        "0-5": 0,
        "5-10": 0,
        "10-20": 0,
        "20-30": 0,
        "30-60": 0,
        "60+": 0,
    }

    cursor.execute(
        f"SELECT total_time FROM rag_metrics {where_clause}",
        params
    )

    for (t,) in cursor.fetchall():
        if t < 5:
            latency_buckets["0-5"] += 1
        elif t < 10:
            latency_buckets["5-10"] += 1
        elif t < 20:
            latency_buckets["10-20"] += 1
        elif t < 30:
            latency_buckets["20-30"] += 1
        elif t < 60:
            latency_buckets["30-60"] += 1
        else:
            latency_buckets["60+"] += 1

    # -------------------------
    # SCORE DISTRIBUTION
    # -------------------------

    score_buckets = {
        "0-0.3": 0,
        "0.3-0.5": 0,
        "0.5-0.7": 0,
        "0.7-1.0": 0,
    }

    cursor.execute(
        f"SELECT max_score FROM rag_metrics {where_clause}",
        params
    )

    for (score,) in cursor.fetchall():
        if score is None:
            continue
        if score < 0.3:
            score_buckets["0-0.3"] += 1
        elif score < 0.5:
            score_buckets["0.3-0.5"] += 1
        elif score < 0.7:
            score_buckets["0.5-0.7"] += 1
        else:
            score_buckets["0.7-1.0"] += 1
    # -------------------------
    # HEALTH SUMMARY
    # -------------------------

    health = []

    if avg_confidence > 0.7:
        health.append("Retrieval работает стабильно (confidence высокий).")
    elif avg_confidence > 0.6:
        health.append("Retrieval удовлетворительный, возможна точечная оптимизация.")
    else:
        health.append("Confidence низкий — требуется улучшение retrieval.")

    if avg_total_time > 25:
        health.append("Время ответа высокое — узкое место LLM.")
    elif avg_total_time > 15:
        health.append("Время ответа среднее.")
    else:
        health.append("Скорость ответа хорошая.")

    if rejection_rate > 0.25:
        health.append("Процент отказов высокий — возможно threshold слишком строгий.")
    elif rejection_rate > 0.15:
        health.append("Процент отказов в пределах нормы.")
    else:
        health.append("Система отвечает почти на все запросы.")
    # -------------------------
    # CONFIDENCE TREND
    # -------------------------

    trend_conditions = []
    trend_params = []

    if from_date and to_date:
        trend_conditions.append("date(created_at) BETWEEN ? AND ?")
        trend_params.extend([from_date, to_date])

    trend_conditions.append("confidence IS NOT NULL")

    trend_where = ""
    if trend_conditions:
        trend_where = "WHERE " + " AND ".join(trend_conditions)

    cursor.execute(f"""
    SELECT date(created_at), AVG(confidence)
    FROM rag_metrics
    {trend_where}
    GROUP BY date(created_at)
    ORDER BY date(created_at)
    """, trend_params)

    rows = cursor.fetchall()

    confidence_trend = [
        {"date": row[0], "value": round(row[1], 3)}
        for row in rows if row[1] is not None
    ]

    # -------------------------
    # TOPIC MODE DISTRIBUTION
    # -------------------------

    cursor.execute(f"""
    SELECT topic_mode, COUNT(*)
    FROM rag_metrics
    {where_clause}
    GROUP BY topic_mode
    """, params)

    topic_distribution = dict(cursor.fetchall())

    # -------------------------
    # FOLLOW-UP RATE
    # -------------------------

    cursor.execute(f"""
    SELECT SUM(is_followup), COUNT(*)
    FROM rag_metrics
    {where_clause}
    """, params)

    followups, total = cursor.fetchone()

    followup_rate = (followups / total) if total else 0

    # -------------------------
    # AVG RETRIEVAL VS LLM TIME
    # -------------------------

    cursor.execute(f"""
    SELECT AVG(retrieval_time), AVG(llm_time)
    FROM rag_metrics
    {where_clause}
    """, params)

    avg_retrieval_time, avg_llm_time = cursor.fetchone()

    # -------------------------
    # ADVANCED ENTERPRISE METRICS
    # -------------------------

    cursor.execute(f"""
    SELECT 
        AVG(max_score),
        AVG(coverage),
        AVG(threshold)
    FROM rag_metrics
    {where_clause}
    """, params)

    row = cursor.fetchone()

    avg_max_score = row[0] if row and row[0] else 0
    avg_coverage = row[1] if row and row[1] else 0
    avg_threshold = row[2] if row and row[2] else 0
    conn.close()
    return JSONResponse({
        "aggregates": {
            "total_requests": total_requests,
            "success_count": success_count,
            "rejection_count": rejection_count,
            "rejection_rate": rejection_rate,
            "avg_confidence": avg_confidence,
            "avg_total_time": avg_total_time,
            "avg_llm_time": avg_llm_time,
            "avg_context_chars": avg_context,
        },

        # --- Основные распределения ---
        "rejection_types": rejection_types,
        "confidence_distribution": confidence_buckets,
        "latency_distribution": latency_buckets,
        "scatter_data": scatter_data,

        # --- Новые аналитические блоки ---
        "confidence_trend": confidence_trend,
        "topic_distribution": topic_distribution,
        "followup_rate": followup_rate,
        "avg_retrieval_time": avg_retrieval_time,
        "avg_llm_time": avg_llm_time,

        # --- Health summary ---
        "health_summary": health,
        "score_distribution": score_buckets,

        "advanced_metrics": {
            "avg_max_score": avg_max_score,
            "avg_coverage": avg_coverage,
            "avg_threshold": avg_threshold
        },
    })


@app.get("/stats")
def serve_stats():
    return FileResponse("app/static/stats.html")