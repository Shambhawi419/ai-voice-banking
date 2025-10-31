import sqlite3
from datetime import datetime

DB_NAME = "memory.db"


def init_db():
    conn = sqlite3.connect(DB_NAME)
    cursor = conn.cursor()

    # User profile table
    cursor.execute("""
        CREATE TABLE IF NOT EXISTS users (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            user_id TEXT UNIQUE,
            name TEXT,
            preferred_language TEXT,
            voice_tone TEXT,
            created_at TEXT
        )
    """)

    # Conversation table
    cursor.execute("""
        CREATE TABLE IF NOT EXISTS conversation (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            user_id TEXT,
            role TEXT,
            message TEXT,
            timestamp TEXT
        )
    """)

    # Session table
    cursor.execute("""
        CREATE TABLE IF NOT EXISTS sessions (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            user_id TEXT,
            session_start TEXT,
            session_end TEXT
        )
    """)

    conn.commit()
    conn.close()


def create_user_profile(user_id, name, language="en", tone="neutral"):
    conn = sqlite3.connect(DB_NAME)
    cursor = conn.cursor()
    cursor.execute("""
        INSERT OR IGNORE INTO users (user_id, name, preferred_language, voice_tone, created_at)
        VALUES (?, ?, ?, ?, ?)
    """, (user_id, name, language, tone, datetime.now().isoformat()))
    conn.commit()
    conn.close()


def get_user_profile(user_id):
    conn = sqlite3.connect(DB_NAME)
    cursor = conn.cursor()
    cursor.execute("""
        SELECT name, preferred_language, voice_tone FROM users WHERE user_id = ?
    """, (user_id,))
    result = cursor.fetchone()
    conn.close()

    if result:
        return {"name": result[0], "language": result[1], "tone": result[2]}
    return None


def update_user_preferences(user_id, language=None, tone=None):
    conn = sqlite3.connect(DB_NAME)
    cursor = conn.cursor()
    if language:
        cursor.execute("UPDATE users SET preferred_language = ? WHERE user_id = ?", (language, user_id))
    if tone:
        cursor.execute("UPDATE users SET voice_tone = ? WHERE user_id = ?", (tone, user_id))
    conn.commit()
    conn.close()


def start_session(user_id):
    conn = sqlite3.connect(DB_NAME)
    cursor = conn.cursor()
    cursor.execute("""
        INSERT INTO sessions (user_id, session_start)
        VALUES (?, ?)
    """, (user_id, datetime.now().isoformat()))
    conn.commit()
    conn.close()


def end_session(user_id):
    conn = sqlite3.connect(DB_NAME)
    cursor = conn.cursor()
    cursor.execute("""
        UPDATE sessions
        SET session_end = ?
        WHERE user_id = ? AND session_end IS NULL
    """, (datetime.now().isoformat(), user_id))
    conn.commit()
    conn.close()


def save_message(user_id, role, message):
    conn = sqlite3.connect(DB_NAME)
    cursor = conn.cursor()
    cursor.execute("""
        INSERT INTO conversation (user_id, role, message, timestamp)
        VALUES (?, ?, ?, ?)
    """, (user_id, role, message, datetime.now().isoformat()))
    conn.commit()
    conn.close()


def get_recent_context(user_id, limit=8):
    conn = sqlite3.connect(DB_NAME)
    cursor = conn.cursor()
    cursor.execute("""
        SELECT role, message FROM conversation
        WHERE user_id = ?
        ORDER BY id DESC LIMIT ?
    """, (user_id, limit))
    rows = cursor.fetchall()
    conn.close()
    rows.reverse()
    return [{"role": r[0], "content": r[1]} for r in rows]
