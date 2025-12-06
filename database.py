import sqlite3
import pandas as pd

DB_NAME = "sentiment.db"

def init_db():
    conn = sqlite3.connect(DB_NAME)
    cur = conn.cursor()
    cur.execute("""
        CREATE TABLE IF NOT EXISTS history (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            text TEXT,
            sentiment TEXT,
            created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
        )
    """)
    conn.commit()
    conn.close()


def save_sentiment(text, sentiment):
    conn = sqlite3.connect(DB_NAME)
    cur = conn.cursor()
    cur.execute("""
        CREATE TABLE IF NOT EXISTS history (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            text TEXT,
            sentiment TEXT,
            created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
        )
    """)
    cur.execute("INSERT INTO history (text, sentiment) VALUES (?, ?)", (text, sentiment))
    conn.commit()
    conn.close()

def load_history(limit=50, offset=0) -> pd.DataFrame:
    conn = sqlite3.connect(DB_NAME)
    query = f"""
        SELECT created_at, text, sentiment
        FROM history
        ORDER BY created_at DESC
        LIMIT {limit} OFFSET {offset}
    """
    df = pd.read_sql_query(query, conn)
    conn.close()
    return df