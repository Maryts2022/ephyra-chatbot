import os
import psycopg2
from dotenv import load_dotenv

load_dotenv()
db_url = os.getenv("DATABASE_URL")

sql_commands = [
    "CREATE EXTENSION IF NOT EXISTS vector;",
    """
    CREATE TABLE IF NOT EXISTS knowledge_base (
        id SERIAL PRIMARY KEY,
        question TEXT,
        answer TEXT,
        embedding vector(1536)
    );
    """
]

try:
    conn = psycopg2.connect(db_url)
    cur = conn.cursor()
    for command in sql_commands:
        cur.execute(command)
    conn.commit()
    cur.close()
    conn.close()
    print("✅ Ο πίνακας δημιουργήθηκε επιτυχώς στο Railway!")
except Exception as e:
    print(f"❌ Σφάλμα: {e}")