from sentence_transformers import SentenceTransformer
import psycopg2

# 1. Φόρτωση μοντέλου
model = SentenceTransformer('all-MiniLM-L6-v2')

# 2. Σύνδεση στη βάση
conn = psycopg2.connect(
    database="chatbot_municipal",
    user="postgres",
    password="1234",
    host="localhost",
    port="5432"
)
cur = conn.cursor()

# 3. Ανάκτηση όλων των εγγραφών (id + κείμενο)
cur.execute("SELECT id, COALESCE(question,'') || ' ' || COALESCE(answer,'') AS text FROM kb_items_raw;")
rows = cur.fetchall()

# 4. Δημιουργία και αποθήκευση embeddings
for _id, text in rows:
    emb = model.encode(text).tolist()
    cur.execute("UPDATE kb_items_raw SET embedding_384 = %s::vector WHERE id = %s;", (emb, _id))

# 5. Αποθήκευση και κλείσιμο
conn.commit()
conn.close()
