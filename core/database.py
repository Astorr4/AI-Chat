import sqlite3
import time
from config import DB_PATH



def get_connection():
    conn = sqlite3.connect(
        DB_PATH,
        timeout=30,
        check_same_thread=False,
        isolation_level=None  # ← ВАЖНО
    )
    conn.row_factory = sqlite3.Row
    conn.execute("PRAGMA busy_timeout=30000;")
    return conn


# ==========================
# SESSION FUNCTIONS
# ==========================

def create_session(session_id):
    with get_connection() as conn:

        conn.execute(
            "INSERT OR IGNORE INTO sessions (id, created_at) VALUES (?, ?)",
            (session_id, time.time())
        )



def save_message(session_id, role, content):
    with get_connection() as conn:

        conn.execute(
            "INSERT INTO messages (session_id, role, content, created_at) VALUES (?, ?, ?, ?)",
            (session_id, role, content, time.time())
        )



def get_last_messages(session_id, limit=4):
    with get_connection() as conn:

        cursor = conn.execute(
            "SELECT role, content FROM messages WHERE session_id = ? ORDER BY id DESC LIMIT ?",
            (session_id, limit)
        )
        rows = cursor.fetchall()

        return list(reversed(rows))


# ==========================
# STATS FUNCTIONS
# ==========================

def update_stats(response_time):
    with get_connection() as conn:

        conn.execute("""
            UPDATE stats
            SET total_requests = total_requests + 1,
                total_response_time = total_response_time + ?
            WHERE id = 1
        """, (response_time,))



def get_stats():
    with get_connection() as conn:

        cursor = conn.execute("SELECT total_requests, total_response_time FROM stats WHERE id = 1")
        row = cursor.fetchone()

        return row


# ==========================
# QUESTION STATS
# ==========================


def get_top_questions(limit=10):
    with get_connection() as conn:
        cursor = conn.execute("""
            SELECT question, count
            FROM question_stats
            ORDER BY count DESC
            LIMIT ?
        """, (limit,))
        rows = cursor.fetchall()
        return rows


def cleanup_old_data(
        days_messages=7,
        days_metrics=30,
        days_question_stats=180
):
    """
    Безопасная очистка старых данных.
    - Сообщения: 7 дней
    - Метрики: 30 дней
    - Статистика вопросов: 180 дней
    """

    with get_connection() as conn:
        cutoff_messages = time.time() - (days_messages * 86400)

        # =========================
        # 1. Удаляем старые сообщения
        # =========================
        conn.execute(
            "DELETE FROM messages WHERE created_at < ?",
            (cutoff_messages,)
        )

        # =========================
        # 2. Удаляем пустые сессии
        # =========================
        conn.execute("""
            DELETE FROM sessions
            WHERE id NOT IN (
                SELECT DISTINCT session_id FROM messages
            )
        """)

        # =========================
        # 3. Очищаем старые RAG метрики
        # =========================
        conn.execute(
            """
            DELETE FROM rag_metrics
            WHERE created_at < datetime('now', ?)
            """,
            (f"-{days_metrics} days",)
        )

        # =========================
        # 4. Чистим старые question_stats (опционально)
        # =========================
        conn.execute(
            """
            DELETE FROM question_stats
            WHERE question NOT IN (
                SELECT question FROM rag_metrics
            )
            """
        )


def save_session(session_id):
    with get_connection() as conn:
        conn.execute(
            "INSERT OR IGNORE INTO sessions (id, created_at) VALUES (?, ?)",
            (session_id, time.time())
        )


def increment_question_stat(question: str):
    with get_connection() as conn:
        cursor = conn.cursor()

        cursor.execute("""
            INSERT INTO question_stats (question, count)
            VALUES (?, 1)
            ON CONFLICT(question)
            DO UPDATE SET count = count + 1
        """, (question,))


def get_messages(session_id):
    with get_connection() as conn:
        cursor = conn.cursor()
        cursor.execute("""
            SELECT role, content
            FROM messages
            WHERE session_id = ?
            ORDER BY created_at ASC
        """, (session_id,))

        rows = cursor.fetchall()

        return [{"role": r[0], "content": r[1]} for r in rows]

def migrate_rag_metrics():
    with get_connection() as conn:
        cursor = conn.cursor()

        cursor.execute("PRAGMA table_info(rag_metrics)")
        columns = [row[1] for row in cursor.fetchall()]

        if "similarity" not in columns:
            cursor.execute(
                "ALTER TABLE rag_metrics ADD COLUMN similarity REAL"
            )

        if "topic_mode" not in columns:
            cursor.execute(
                "ALTER TABLE rag_metrics ADD COLUMN topic_mode TEXT"
            )

        if "max_score" not in columns:
            cursor.execute("ALTER TABLE rag_metrics ADD COLUMN max_score REAL")

        if "coverage" not in columns:
            cursor.execute("ALTER TABLE rag_metrics ADD COLUMN coverage REAL")

        if "threshold" not in columns:
            cursor.execute("ALTER TABLE rag_metrics ADD COLUMN threshold REAL")


def init_db():
    with get_connection() as conn:

        cursor = conn.cursor()

        # WAL режим для конкурентности
        conn.execute("PRAGMA journal_mode=WAL;")
        conn.execute("PRAGMA synchronous=NORMAL;")

        # ==========================
        # SESSIONS
        # ==========================
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS sessions (
                id TEXT PRIMARY KEY,
                created_at REAL
            )
        """)

        # ==========================
        # MESSAGES
        # ==========================
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS messages (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                session_id TEXT,
                role TEXT,
                content TEXT,
                created_at REAL
            )
        """)

        # ==========================
        # GLOBAL STATS
        # ==========================
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS stats (
                id INTEGER PRIMARY KEY CHECK (id = 1),
                total_requests INTEGER,
                total_response_time REAL
            )
        """)

        cursor.execute("""
            INSERT OR IGNORE INTO stats (id, total_requests, total_response_time)
            VALUES (1, 0, 0)
        """)

        # ==========================
        # QUESTION STATS
        # ==========================
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS question_stats (
                question TEXT PRIMARY KEY,
                count INTEGER
            )
        """)

        # ==========================
        # RAG METRICS
        # ==========================
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS rag_metrics (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                session_id TEXT,
                user_login TEXT,
                question TEXT,
                found_docs INTEGER,
                filtered_docs INTEGER,
                confidence REAL,
                avg_score REAL,
                max_score REAL,
                coverage REAL,
                threshold REAL,
                min_score REAL,
                sources_count INTEGER,
                context_chars INTEGER,
                retrieval_time REAL,
                llm_time REAL,
                total_time REAL,
                answer_length INTEGER,
                is_followup INTEGER,
                memory_size INTEGER,
                rejected_reason TEXT,
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
            )
        """)


    # 🔥 Авто-миграция (если таблица старая)
    migrate_rag_metrics()

def save_rag_metrics(
        session_id,
        user_login,
        question,
        found_docs,
        filtered_docs,
        confidence,
        avg_score,
        min_score,
        max_score,
        coverage,
        threshold,
        sources_count,
        context_chars,
        retrieval_time,
        llm_time,
        total_time,
        answer_length,
        is_followup,
        memory_size,
        rejected_reason=None,
        similarity=None,
        topic_mode=None
):
    with get_connection() as conn:

        cursor = conn.cursor()

        cursor.execute("""
        INSERT INTO rag_metrics (
            session_id,
            user_login,
            question,
            found_docs,
            filtered_docs,
            confidence,
            avg_score,
            min_score,
            max_score,
            coverage,
            threshold,
            sources_count,
            context_chars,
            retrieval_time,
            llm_time,
            total_time,
            answer_length,
            is_followup,
            memory_size,
            rejected_reason,
            similarity,
            topic_mode
        ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        """, (
                session_id,
                user_login,
                question,
                found_docs,
                filtered_docs,
                confidence,
                avg_score,
                min_score,
                max_score,
                coverage,
                threshold,
                sources_count,
                context_chars,
                retrieval_time,
                llm_time,
                total_time,
                answer_length,
                is_followup,
                memory_size,
                rejected_reason,
                similarity,
                topic_mode
            ))

