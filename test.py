import sqlite3

DB_PATH = "ai.db"  # ⚠ если у тебя другое имя БД — поменяй

with sqlite3.connect(DB_PATH) as conn:
    cursor = conn.cursor()

    cursor.execute("PRAGMA table_info(rag_metrics)")
    existing_columns = {row[1] for row in cursor.fetchall()}

    if "user_login" not in existing_columns:
        cursor.execute("ALTER TABLE rag_metrics ADD COLUMN user_login TEXT")
