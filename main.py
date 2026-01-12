"""
Ephyra Chatbot - Production RAG (Retrieval Augmented Generation)
Final Optimized Version - Full Features (Chat, Survey, Dashboard, TTS)
"""

import os
import io
import uuid
import logging
import csv
import string
import json
import re
from functools import lru_cache
from pathlib import Path
from typing import Optional, List, Dict, Tuple
from datetime import datetime, timedelta

# Third-party imports
import psycopg2
import psycopg2.pool
from fastapi import FastAPI, HTTPException, Request, BackgroundTasks
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import StreamingResponse, FileResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel
from dotenv import load_dotenv
from slowapi import Limiter, _rate_limit_exceeded_handler
from slowapi.util import get_remote_address
from slowapi.errors import RateLimitExceeded
from elevenlabs.client import ElevenLabs
from sentence_transformers import SentenceTransformer
from openai import OpenAI

# ================== 1. Configuration & Setup ==================

# Load Environment Variables
load_dotenv(dotenv_path=Path(__file__).with_name(".env"), override=True)

# Logging Setup
logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
log = logging.getLogger("ephyra")

# Check Required Vars
required_vars = ["OPENAI_API_KEY", "DB_NAME", "DB_USER", "DB_PASS", "DB_HOST"]
missing = [v for v in required_vars if not os.getenv(v)]
if missing:
    raise RuntimeError(f"Missing env vars: {', '.join(missing)}")

# Initialize Clients
openai_client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))

ELEVENLABS_API_KEY = (os.getenv("ELEVENLABS_API_KEY") or "").strip()
ELEVENLABS_VOICE_ID = (os.getenv("ELEVENLABS_VOICE_ID") or "EXAVITQu4vr4xnSDxMaL").strip()
eleven_client = ElevenLabs(api_key=ELEVENLABS_API_KEY)

# Load Knowledge Base (CSV) into Memory (Backup)
knowledge_base = []
try:
    with open("QA_chatbot.csv", mode="r", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            knowledge_base.append(row)
except Exception as e:
    log.warning(f"⚠️ Could not load QA_chatbot.csv locally: {e}")

# ================== 2. Database Connection Pool ==================

try:
    conn_pool = psycopg2.pool.SimpleConnectionPool(
        5, 20,
        dbname=os.getenv("DB_NAME"),
        user=os.getenv("DB_USER"),
        password=os.getenv("DB_PASS"),
        host=os.getenv("DB_HOST"),
        port=os.getenv("DB_PORT", "5432"),
    )
    log.info("✅ Database connection pool created")
except Exception as e:
    log.exception("❌ Database connection failed")
    raise

def get_db_conn():
    try:
        return conn_pool.getconn()
    except Exception as e:
        log.exception("❌ Failed to get DB connection from pool")
        raise

def return_db_conn(conn):
    if conn:
        try:
            conn_pool.putconn(conn)
        except Exception as e:
            log.error(f"❌ Error returning connection to pool: {e}")

# ================== 3. Database Initialization (Tables) ==================

def init_all_tables():
    """Initialize all necessary tables (Knowledge Base, Feedback, Survey)."""
    conn = get_db_conn()
    cur = conn.cursor()
    try:
        # A. Knowledge Base Table
        cur.execute("CREATE EXTENSION IF NOT EXISTS vector;")
        cur.execute("""
            CREATE TABLE IF NOT EXISTS public.kb_items_raw (
                id SERIAL PRIMARY KEY,
                question TEXT,
                answer TEXT,
                category TEXT,
                embedding_384 vector(384)
            );
        """)

        # B. Feedback Table
        cur.execute("""
            CREATE TABLE IF NOT EXISTS chatbot_feedback (
                id SERIAL PRIMARY KEY,
                conversation_id TEXT,
                bot_response TEXT,
                user_question TEXT,
                is_positive BOOLEAN,
                timestamp TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                user_agent TEXT,
                ip_address TEXT
            );
        """)

        # C. Survey Table (Re-create to ensure schema match)
        cur.execute("""
            CREATE TABLE IF NOT EXISTS survey_final (
                id SERIAL PRIMARY KEY,
                timestamp TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                used_bot TEXT,
                usage_context TEXT,
                scenarios_tested TEXT,
                gender TEXT,
                age TEXT,
                q1 INTEGER, q2 INTEGER, q3 INTEGER, q4 INTEGER, q5 INTEGER,
                q6 INTEGER, q7 INTEGER, q8 INTEGER, q9 INTEGER, q10 INTEGER,
                q11 INTEGER, q12 INTEGER, q13 INTEGER, q14 INTEGER, q15 INTEGER,
                q16 INTEGER,
                comments TEXT
            );
        """)
        
        conn.commit()
        log.info("✅ All database tables checked/initialized.")
    except Exception as e:
        log.error(f"❌ Error initializing tables: {e}")
        if conn: conn.rollback()
    finally:
        cur.close()
        return_db_conn(conn)

# ================== 4. Background Tasks (CSV Sync) ==================

def sync_csv_to_db():
    """Reads CSV and updates the Vector Database."""
    try:
        conn = get_db_conn()
        cur = conn.cursor()

        log.info("🔄 Syncing CSV to DB...")
        model = SentenceTransformer('paraphrase-multilingual-MiniLM-L12-v2')

        # Clean old data & Re-index
        cur.execute("TRUNCATE public.kb_items_raw;")
        cur.execute("""
            CREATE INDEX IF NOT EXISTS kb_items_embedding_idx 
            ON public.kb_items_raw 
            USING hnsw (embedding_384 vector_cosine_ops);
        """)

        with open("QA_chatbot.csv", mode="r", encoding="utf-8") as f:
            reader = csv.DictReader(f)
            count = 0
            for row in reader:
                values = list(row.values())
                if len(values) >= 2:
                    q, a = values[0], values[1]
                    emb = model.encode(q).tolist()
                    cur.execute(
                        "INSERT INTO kb_items_raw (question, answer, embedding_384) VALUES (%s, %s, %s)",
                        (q, a, emb)
                    )
                    count += 1
        
        conn.commit()
        log.info(f"✅ Database sync complete! Loaded {count} items.")
    except Exception as e:
        log.error(f"❌ Sync failed: {e}")
        if conn: conn.rollback()
    finally:
        if cur: cur.close()
        if conn: return_db_conn(conn)

# ================== 5. Helper Functions & Logic ==================

# Embedder Lazy Load
embedder = None
@lru_cache(maxsize=1)
def get_embedder():
    global embedder
    if embedder is None:
        log.info("📄 Loading SentenceTransformer...")
        embedder = SentenceTransformer("paraphrase-multilingual-MiniLM-L12-v2")
    return embedder

def get_direct_answer(question: str) -> Optional[Dict]:
    """Returns hardcoded answers for critical questions (Cheat Sheet)."""
    text_lower = question.lower().strip()
    
    # 1. ΑΝΤΙΔΗΜΑΡΧΟΙ (Fixed List)
    if 'αντιδήμαρχ' in text_lower or 'αντιδημαρχ' in text_lower:
        # Ειδικά για Καθαριότητα
        if 'καθαριότ' in text_lower or 'καθαριοτ' in text_lower:
             return {
                "answer": "Αντιδήμαρχος Καθαριότητας είναι ο κ. Δημήτριος Μανωλάκης. Τηλέφωνο επικοινωνίας: 2741361000",
                "quality": "direct_match"
            }
        
        # Γενικά για Αντιδημάρχους
        return {
            "answer": """Οι Αντιδήμαρχοι του Δήμου Κορινθίων είναι:

1. Γεώργιος Πούρος (Διοικητικών Υπηρεσιών)
2. Βασίλειος Πανταζής (Πολεοδομίας)
3. Δημήτριος Μανωλάκης (Καθαριότητας)
4. Ευάγγελος Παπαϊωάννου (Παιδείας & Τουρισμού)
5. Ανδρέας Ζώγκος (Τεχνικών Υπηρεσιών)
6. Αναστάσιος Ταγαράς (Πολιτισμού)

Για περισσότερες πληροφορίες καλέστε στο 2741361000.""",
            "quality": "direct_match"
        }

    # 2. ΚΕΠ
    if any(kw in text_lower for kw in ['κεπ', 'κέντρο εξυπηρέτησης πολιτών']):
        return {
            "answer": """ΚΕΠ Κορίνθου - Στοιχεία Επικοινωνίας

📍 Διεύθυνση: Κωστή Παλαμά 53, 20131 Κόρινθος
📞 Τηλέφωνο: 2741363555
📧 Email: n.korinthias@kep.gov.gr
🕒 Ωράριο: Δευτέρα - Παρασκευή 8:00-15:00""",
            "quality": "direct_match"
        }
    
    # 3. ΔΕΥΑ
    if any(kw in text_lower for kw in ['δευα', 'δ.ε.υ.α', 'νερό']):
        return {
            "answer": """ΔΕΥΑ Κορίνθου
📞 Τηλέφωνο Κέντρο: 2741024444
📞 Βλάβες (24ωρο): 6936776041
📧 Email: info@deyakor.gr""",
            "quality": "direct_match"
        }
    
    # 4. Δήμαρχος
    if any(kw in text_lower for kw in ['δήμαρχ', 'δημαρχ', 'mayor']):
        return {
            "answer": """Γραφείο Δημάρχου Κορινθίων

Δήμαρχος: Νίκος Σταυρέλης
📞 Τηλέφωνο: 27413-61001, 27413-61041
📧 Email: grafeiodimarxou@korinthos.gr
📍 Διεύθυνση: Κολιάτσου 32, 201 31 Κόρινθος""",
            "quality": "direct_match"
        }

    return None

def retrieve_context(cursor, question: str, top_k: int = 5) -> List[Dict]:
    """Retrieves relevant context from DB using Semantic Search."""
    try:
        q_embedding = get_embedder().encode(question).tolist()
        cursor.execute("""
            SELECT id, question, answer, 1 - (embedding_384 <=> %s::vector) as similarity
            FROM public.kb_items_raw 
            ORDER BY embedding_384 <-> %s::vector
            LIMIT %s
        """, (q_embedding, q_embedding, top_k))
        
        results = []
        for r in cursor.fetchall():
            results.append({"question": r[1], "answer": r[2], "similarity": float(r[3])})
        return results
    except Exception as e:
        log.error(f"Search Error: {e}")
        return []

# ================== 6. FastAPI App & Middleware ==================

app = FastAPI(title="Ephyra Chatbot - Production RAG", version="3.1.0")

# Mount Static
try:
    static_dir = os.path.dirname(os.path.abspath(__file__))
    app.mount("/static", StaticFiles(directory=static_dir, check_dir=True), name="static")
except Exception as e:
    log.warning(f"⚠️ Could not mount static files: {e}")

# CORS & Rate Limiting
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)
limiter = Limiter(key_func=get_remote_address)
app.state.limiter = limiter
app.add_exception_handler(RateLimitExceeded, _rate_limit_exceeded_handler)

# ================== 7. Pydantic Models ==================

class Message(BaseModel):
    role: str
    content: str

class AskBody(BaseModel):
    messages: List[Message]
    lang: str = "el"

class TTSBody(BaseModel):
    text: str

class SurveyResponse(BaseModel):
    usedBot: str
    usageContext: str
    scenarios: str
    gender: Optional[str] = "N/A"
    age: Optional[str] = "N/A"
    q1: int; q2: int; q3: int; q4: int; q5: int
    q6: int; q7: int; q8: int; q9: int; q10: int
    q11: int; q12: int; q13: int; q14: int; q15: int
    q16: int
    comments: Optional[str] = ""

# ================== 8. Endpoints ==================

@app.get("/")
async def root(background_tasks: BackgroundTasks):
    # Initialize DB & Sync on startup visit
    init_all_tables()
    background_tasks.add_task(sync_csv_to_db) 
    
    current_dir = os.path.dirname(os.path.abspath(__file__))
    html_path = os.path.join(current_dir, "ui_chatbot.html")
    if os.path.exists(html_path):
        return FileResponse(html_path, media_type="text/html")
    return {"message": "Ephyra is online and syncing!"}

@app.get("/health")
async def health():
    return {"status": "healthy"}

@app.get("/dashboard")
async def get_dashboard():
    path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "feedback_dashboard.html")
    if os.path.exists(path): return FileResponse(path)
    return {"error": "Dashboard not found"}

@app.get("/questionnaire")
async def get_questionnaire():
    path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "questionnaire.html")
    if os.path.exists(path): return FileResponse(path)
    return {"error": "Questionnaire not found"}

# --- MAIN CHAT ENDPOINT ---
@app.post("/ask")
@limiter.limit("30/minute")
async def ask(request: Request, body: AskBody):
    current_lang = body.lang or "el"
    question = (body.messages[-1].content if body.messages else "").strip()
    if not question: return {"answer": "..."}

    # 1. ΕΛΕΓΧΟΣ ΓΙΑ ΑΜΕΣΗ ΑΠΑΝΤΗΣΗ (Cheat Sheet)
    # Αυτό λύνει το πρόβλημα με τους Αντιδημάρχους!
    direct_resp = get_direct_answer(question)
    if direct_resp:
        async def direct_stream():
            yield direct_resp["answer"]
        return StreamingResponse(direct_stream(), media_type="text/plain")

    # 2. Αλλιώς, προετοιμασία RAG (CSV + DB)
    csv_context = ""
    def clean_text(t):
        if not t: return ""
        return t.lower().translate(str.maketrans('', '', string.punctuation)).strip()

    clean_user_q = clean_text(question)
    
    # Γρήγορο ψάξιμο στη μνήμη (CSV)
    for row in knowledge_base:
        if len(row) >= 2:
            q_raw, a_val = list(row.values())[0], list(row.values())[1]
            if clean_text(str(q_raw)) in clean_user_q:
                csv_context += f"\nCSV Info: {a_val}\n"

    # 3. Βαθύ ψάξιμο στη Βάση & Γέννηση απάντησης
    async def event_generator():
        conn = get_db_conn()
        try:
            cursor = conn.cursor()
            db_docs = retrieve_context(cursor, question, top_k=4)
            db_text = "\n".join([f"Info: {d['question']} - {d['answer']}" for d in db_docs])
            cursor.close()

            all_context = csv_context + "\n" + db_text

            # System Prompt
            sys_msg = (
                f"Είσαι η Εφύρα, ψηφιακή βοηθός του Δήμου Κορινθίων. "
                f"Απάντησε στη γλώσσα: {current_lang}. "
                f"Χρησιμοποίησε ΜΟΝΟ το παρακάτω CONTEXT."
            )

            response = openai_client.chat.completions.create(
                model="gpt-4o-mini",
                messages=[
                    {"role": "system", "content": sys_msg},
                    {"role": "system", "content": f"CONTEXT:\n{all_context}"},
                    {"role": "user", "content": question}
                ],
                temperature=0.7,
                stream=True
            )
            for chunk in response:
                if chunk.choices[0].delta.content:
                    yield chunk.choices[0].delta.content

        except Exception as e:
            log.error(f"Stream Error: {e}")
            yield "Συγγνώμη, υπήρξε πρόβλημα στη σύνδεση."
        finally:
            return_db_conn(conn)

    return StreamingResponse(event_generator(), media_type="text/plain")

# --- FEEDBACK & SURVEY ENDPOINTS ---

@app.post("/feedback")
async def record_feedback(request: Request):
    try:
        data = await request.json()
        conn = get_db_conn(); cur = conn.cursor()
        
        cur.execute("INSERT INTO chatbot_feedback (conversation_id, bot_response, user_question, is_positive, user_agent, ip_address) VALUES (%s, %s, %s, %s, %s, %s)",
            (data.get("conversation_id"), data.get("bot_response"), data.get("user_question"), data.get("is_positive"), request.headers.get("User-Agent"), request.client.host))
        
        conn.commit(); cur.close(); return_db_conn(conn)
        return {"status": "success"}
    except Exception as e:
        return {"error": str(e)}

@app.get("/feedback/stats")
async def get_feedback_stats(days: int = 30):
    """Provides statistics for the Feedback Dashboard."""
    conn = get_db_conn()
    cur = conn.cursor()
    try:
        since_date = datetime.now() - timedelta(days=days)
        
        # 1. Total Stats
        cur.execute("""
            SELECT 
                COUNT(*) as total,
                SUM(CASE WHEN is_positive THEN 1 ELSE 0 END) as positive,
                SUM(CASE WHEN is_positive THEN 0 ELSE 1 END) as negative
            FROM chatbot_feedback
            WHERE timestamp >= %s
        """, (since_date,))
        result = cur.fetchone()
        total = result[0] or 0
        positive = result[1] or 0
        negative = result[2] or 0
        satisfaction_rate = round((positive / total * 100)) if total > 0 else 0

        # 2. Daily Data
        cur.execute("""
            SELECT DATE(timestamp) as date, is_positive, COUNT(*)
            FROM chatbot_feedback
            WHERE timestamp >= %s
            GROUP BY DATE(timestamp), is_positive
            ORDER BY date
        """, (since_date,))
        daily_data = []
        for row in cur.fetchall():
            daily_data.append({
                "date": str(row[0]),
                "sentiment": "positive" if row[1] else "negative",
                "count": row[2]
            })

        # 3. Recent Feedback
        cur.execute("""
            SELECT user_question, bot_response, is_positive, timestamp
            FROM chatbot_feedback
            WHERE timestamp >= %s
            ORDER BY timestamp DESC LIMIT 20
        """, (since_date,))
        recent = []
        for row in cur.fetchall():
            recent.append({
                "user_question": row[0],
                "bot_response": row[1],
                "is_positive": row[2],
                "timestamp": str(row[3])
            })

        return {
            "total_feedback": total,
            "positive": positive,
            "negative": negative,
            "satisfaction_rate": satisfaction_rate,
            "daily_data": daily_data,
            "recent_feedback": recent
        }
    except Exception as e:
        log.error(f"Stats Error: {e}")
        return {"error": str(e)}
    finally:
        cur.close()
        return_db_conn(conn)

@app.post("/submit_survey")
async def submit_survey(data: SurveyResponse):
    try:
        conn = get_db_conn(); cur = conn.cursor()
        
        cur.execute("""
            INSERT INTO survey_final 
            (used_bot, usage_context, scenarios_tested, gender, age, 
             q1, q2, q3, q4, q5, q6, q7, q8, q9, q10, 
             q11, q12, q13, q14, q15, q16, comments)
            VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s)
        """, (data.usedBot, data.usageContext, data.scenarios, data.gender, data.age, data.q1, data.q2, data.q3, data.q4, data.q5, data.q6, data.q7, data.q8, data.q9, data.q10, data.q11, data.q12, data.q13, data.q14, data.q15, data.q16, data.comments))
        
        conn.commit(); cur.close(); return_db_conn(conn)
        return {"status": "success"}
    except Exception as e:
        return {"error": str(e)}

@app.get("/survey_results")
async def get_survey_results():
    try:
        conn = get_db_conn(); cur = conn.cursor()
        cur.execute("SELECT * FROM survey_final ORDER BY timestamp DESC")
        cols = [desc[0] for desc in cur.description]
        results = [dict(zip(cols, row)) for row in cur.fetchall()]
        for r in results:
            if 'timestamp' in r and r['timestamp']: r['timestamp'] = str(r['timestamp'])
        cur.close(); return_db_conn(conn)
        return results
    except: return []

# --- TTS ENDPOINTS ---
@app.get("/tts_play")
@app.post("/tts_play")
async def tts_play(request: Request, text: str = "", body: TTSBody = None):
    # Handle both GET and POST
    final_text = text
    if body and body.text:
        final_text = body.text
        
    if not final_text: return HTTPException(400)
    
    try:
        clean = final_text.replace("📍","").replace("📞","").strip()
        # Regex to space out numbers for better reading (2 7 4 1 ...)
        clean = re.sub(r'(\d)', r'\1 ', clean)
        
        audio = eleven_client.text_to_speech.convert(
            voice_id=ELEVENLABS_VOICE_ID,
            text=clean, model_id="eleven_multilingual_v2"
        )
        data = b"".join(chunk for chunk in audio if chunk)
        return StreamingResponse(io.BytesIO(data), media_type="audio/mpeg")
    except: return HTTPException(500)

# 👇 ΕΔΩ ΜΠΑΙΝΕΙ Ο ΝΕΟΣ ΚΩΔΙΚΑΣ 👇
@app.post("/feedback/clear")
async def clear_all_data():
    """Wipes BOTH Feedback and Survey tables for a clean slate."""
    conn = get_db_conn()
    try:
        cur = conn.cursor()
        # 1. Διαγραφή Chat Feedback
        cur.execute("TRUNCATE TABLE chatbot_feedback;")
        # 2. Διαγραφή Ερωτηματολογίων (Survey)
        cur.execute("TRUNCATE TABLE survey_final;")
        conn.commit()
        cur.close()
        return {"status": "success", "message": "All data wiped successfully"}
    except Exception as e:
        log.error(f"Clear Error: {e}")
        return {"error": str(e)}
    finally:
        return_db_conn(conn)
# 👆 ΤΕΛΟΣ ΝΕΟΥ ΚΩΔΙΚΑ 👆 


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)