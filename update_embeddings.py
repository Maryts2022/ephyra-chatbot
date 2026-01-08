import os
import psycopg2
from sentence_transformers import SentenceTransformer
from dotenv import load_dotenv
from pathlib import Path

# Load .env
load_dotenv(dotenv_path=Path(__file__).with_name(".env"))

# Connect to DB
conn = psycopg2.connect(
    dbname=os.getenv("DB_NAME"),
    user=os.getenv("DB_USER"),
    password=os.getenv("DB_PASS"),
    host=os.getenv("DB_HOST"),
    port=os.getenv("DB_PORT"),
)
cur = conn.cursor()

# Load NEW model (το ίδιο με το API)
print("Loading model...")
model = SentenceTransformer('paraphrase-multilingual-MiniLM-L12-v2')

# Get all questions
print("Fetching questions...")
cur.execute("SELECT id, question FROM kb_items_raw;")
rows = cur.fetchall()
print(f"Found {len(rows)} questions")

# Update embeddings
for i, (id, question) in enumerate(rows):
    if i % 10 == 0:
        print(f"Processing {i}/{len(rows)}...")
    
    # Create new embedding
    embedding = model.encode(question).tolist()
    
    # Update in DB
    cur.execute(
        "UPDATE kb_items_raw SET embedding_384 = %s::vector WHERE id = %s;",
        (embedding, id)
    )

# Commit changes
conn.commit()
print("✅ Done! All embeddings updated!")

cur.close()
conn.close()