import uuid
import asyncio
import time
import logging
from collections import defaultdict, deque
import getpass

from fastapi import FastAPI, Request
from fastapi.staticfiles import StaticFiles
from fastapi.responses import FileResponse, StreamingResponse, JSONResponse
from pydantic import BaseModel
from core.embeddings import EmbeddingModel
from core.vector_store import VectorStore
from core.llm import LLM
from core.rag import RAG
from app.services.chat_service import ChatService
from app.services.stats_service import build_rag_stats
from core.database import (
    init_db,
    save_session,
    save_message,
    increment_question_stat,
    cleanup_old_data,
    save_rag_metrics,
)
from config import (
    RATE_LIMIT,
    RATE_WINDOW,
    LLM_CONCURRENCY_LIMIT,
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
chat_service = ChatService(rag=rag, llm_semaphore=llm_semaphore, logger=logger)

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
    response = StreamingResponse(
        chat_service.stream_chat(
            question=query.question,
            session_id=session_id,
            user_login=user_login,
        ),
        media_type="text/plain",
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

    result = rag.search(cleaned_question, top_k=3, session_id=session_id)

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
    data = build_rag_stats(from_date=from_date, to_date=to_date)
    return JSONResponse(data)


@app.get("/stats")
def serve_stats():
    return FileResponse("app/static/stats.html")
