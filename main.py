"""
Ephyra Chatbot - Production RAG (Retrieval Augmented Generation)
LLM-Based Architecture with Semantic Search, Keyword Search, and Gemini Generation
"""

from functools import lru_cache
import os
import io
import uuid
import logging
from pathlib import Path
from typing import AsyncGenerator, Optional, List, Dict, Tuple
from datetime import datetime, timedelta
import json
import psycopg2
import psycopg2.pool
import httpx
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
import re
from sentence_transformers import SentenceTransformer
from langdetect import detect, LangDetectException
from fastapi.middleware.cors import CORSMiddleware
import csv
import string

 
# Δημιουργούμε μια λίστα για να αποθηκεύσουμε τις ερωταπαντήσεις
knowledge_base = []

# Ανοίγουμε το αρχείο που ανέβασες στο GitHub
with open("QA_chatbot.csv", mode="r", encoding="utf-8") as f:
    reader = csv.DictReader(f)
    for row in reader:
        knowledge_base.append(row)


# ================== Configuration ==================
load_dotenv(dotenv_path=Path(__file__).with_name(".env"), override=True)





logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
log = logging.getLogger("ephyra")

required_vars = ["OPENAI_API_KEY", "DB_NAME", "DB_USER", "DB_PASS", "DB_HOST"]
missing = [v for v in required_vars if not os.getenv(v)]
if missing:
    raise RuntimeError(f"Missing env vars: {', '.join(missing)}")

# Εδώ προσθέτεις την αρχικοποίηση του OpenAI
from openai import OpenAI
openai_client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))
# ================== FastAPI Setup ==================
app = FastAPI(title="Ephyra Chatbot - Production RAG", version="3.0.0")
app.mount("/static", StaticFiles(directory="static"), name="static")
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

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# ================== AUTO-SYNC CSV TO DATABASE ==================
def sync_csv_to_db():
    try:
        conn = psycopg2.connect(
            dbname=os.getenv("DB_NAME"),
            user=os.getenv("DB_USER"),
            password=os.getenv("DB_PASS"),
            host=os.getenv("DB_HOST"),
            port=os.getenv("DB_PORT", "5432")
        )
        cur = conn.cursor()

        # 1. Δημιουργία πίνακα
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

        # 2. Φόρτωση μοντέλου
        log.info("🔄 Syncing CSV to DB...")
        model = SentenceTransformer('paraphrase-multilingual-MiniLM-L12-v2')

        # 3. Καθαρισμός & ΔΗΜΙΟΥΡΓΙΑ ΕΥΡΕΤΗΡΙΟΥ (HNSW) 🚀
        cur.execute("TRUNCATE public.kb_items_raw;")
        cur.execute("""
            CREATE INDEX IF NOT EXISTS kb_items_embedding_idx 
            ON public.kb_items_raw 
            USING hnsw (embedding_384 vector_cosine_ops);
        """)

        # 4. Ανέβασμα από CSV
        with open("QA_chatbot.csv", mode="r", encoding="utf-8") as f:
            reader = csv.DictReader(f)
            for row in reader:
                values = list(row.values())
                if len(values) >= 2:
                    q, a = values[0], values[1]
                    emb = model.encode(q).tolist()
                    cur.execute(
                        "INSERT INTO kb_items_raw (question, answer, embedding_384) VALUES (%s, %s, %s)",
                        (q, a, emb)
                    )
        
        conn.commit()
        cur.close()
        conn.close()
        log.info("✅ Database sync complete & Index created!")
    except Exception as e:
        log.error(f"❌ Sync failed: {e}")


# ===============================================================

# Mount static files
try:
    static_dir = os.path.dirname(os.path.abspath(__file__))
    app.mount("/static", StaticFiles(directory=static_dir, check_dir=True), name="static")
except Exception as e:
    log.warning(f"⚠️ Could not mount static files: {e}")

# ================== Database ==================
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

# 1. Η βασική συνάρτηση
def get_db_conn():
    try:
        return conn_pool.getconn()
    except Exception as e:
        log.exception("❌ Failed to get DB connection from pool")
        raise

# 2. Η συνάρτηση επιστροφής
def return_db_conn(conn):
    if conn:
        try:
            conn_pool.putconn(conn)
        except Exception as e:
            log.error(f"❌ Error returning connection to pool: {e}")

class SurveyResponse(BaseModel):
    usedBot: str
    usageContext: str
    scenarios: str
    gender: Optional[str] = "N/A"  # Προσθήκη
    age: Optional[str] = "N/A"     # Προσθήκη
    q1: int; q2: int; q3: int; q4: int; q5: int
    q6: int; q7: int; q8: int; q9: int; q10: int
    q11: int; q12: int; q13: int; q14: int; q15: int
    q16: int                       # Προσθήκη
    comments: Optional[str] = ""


# --- AYTOMATH ΔΗΜΙΟΥΡΓΙΑ ΠΙΝΑΚΑ SURVEY ---
def init_survey_db():
    conn = get_db_conn() 
    cur = conn.cursor()
    try:
        # Διαγράφουμε τον πίνακα για να εφαρμοστούν οι νέες στήλες σωστά
        cur.execute("DROP TABLE IF EXISTS survey_final CASCADE;") 

        cur.execute("""
            CREATE TABLE IF NOT EXISTS survey_final (
                id SERIAL PRIMARY KEY,
                timestamp TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                used_bot TEXT,
                usage_context TEXT,
                scenarios_tested TEXT, -- Η στήλη για τα σενάρια
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
    except Exception as e:
        log.error(f"❌ Error initializing survey table: {e}")
    finally:
        cur.close()
        return_db_conn(conn)

init_survey_db()

@app.post("/submit_survey")
async def submit_survey(data: SurveyResponse):
    conn = get_db_conn()
    cur = conn.cursor()
    try:
        # 22 στήλες συνολικά
        query = """
            INSERT INTO survey_final 
            (used_bot, usage_context, scenarios_tested, gender, age, 
             q1, q2, q3, q4, q5, q6, q7, q8, q9, q10, 
             q11, q12, q13, q14, q15, q16, comments)
            VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s)
        """
        
        cur.execute(query, (
            data.usedBot,           # used_bot
            data.usageContext,      # usage_context
            data.scenarios,         # scenarios_tested (ΕΔΩ ΜΠΑΙΝΟΥΝ ΤΑ ΣΕΝΑΡΙΑ)
            data.gender,            # gender
            data.age,               # age
            data.q1, data.q2, data.q3, data.q4, data.q5,
            data.q6, data.q7, data.q8, data.q9, data.q10,
            data.q11, data.q12, data.q13, data.q14, data.q15,
            data.q16,               # q16
            data.comments           # comments
        ))
        conn.commit()
        return {"status": "success"}
    except Exception as e:
        if conn: conn.rollback()
        log.error(f"❌ Database Insertion Error: {e}")
        return {"status": "error", "message": str(e)}
    finally:
        cur.close()
        return_db_conn(conn)

 
# 3. Τα Aliases (για να μη χτυπάει πουθενά ο κώδικας)
get_db_connection = get_db_conn
return_db_connection = return_db_conn

# 4. Αυτόματη δημιουργία πίνακα feedback αν λείπει
def init_feedback_table():
    conn = None
    try:
        conn = get_db_conn()
        cursor = conn.cursor()
        
        # ΠΡΟΣΟΧΗ: Διαγράφουμε τον παλιό πίνακα για να τον ξαναφτιάξουμε σωστά
        #cursor.execute("DROP TABLE IF EXISTS chatbot_feedback CASCADE;") 
        
        # Δημιουργία με τα ονόματα που θέλουν οι συναρτήσεις σου (timestamp και ip_address)
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS chatbot_feedback (
                id SERIAL PRIMARY KEY,
                conversation_id TEXT,
                bot_response TEXT,
                user_question TEXT,
                is_positive BOOLEAN,
                timestamp TIMESTAMP DEFAULT CURRENT_TIMESTAMP, -- Ονομάστηκε timestamp για το Dashboard
                user_agent TEXT,
                ip_address TEXT  -- Ονομάστηκε ip_address για το record_feedback
            );
        """)
        conn.commit()
        log.info("🚀 Database Table 'chatbot_feedback' is now PERFECT!")
    except Exception as e:
        log.error(f"❌ Error initializing table: {e}")
    finally:
        if conn:
            return_db_conn(conn)

init_feedback_table()

# ================== Embeddings (Lazy Load) ==================
embedder = None

@lru_cache(maxsize=1)
def get_embedder():
    global embedder
    if embedder is None:
        log.info("📄 Loading SentenceTransformer (first use)...")
        embedder = SentenceTransformer("paraphrase-multilingual-MiniLM-L12-v2")
        log.info("✅ SentenceTransformer loaded")
    return embedder

# ================== ElevenLabs Setup ==================
from elevenlabs.client import ElevenLabs

ELEVENLABS_API_KEY = (os.getenv("ELEVENLABS_API_KEY") or "").strip()
ELEVENLABS_VOICE_ID = (os.getenv("ELEVENLABS_VOICE_ID") or "EXAVITQu4vr4xnSDxMaL").strip()
eleven_client = ElevenLabs(api_key=ELEVENLABS_API_KEY)
# ================== Pydantic Models ==================
class Message(BaseModel):
    role: str
    content: str

class AskBody(BaseModel):
    messages: List[Message]
    top_k: int = 10  # Changed from 5 to 10 for better coverage
    lang: str = "el"

class TTSBody(BaseModel):
    text: str

class FeedbackBody(BaseModel):
    bot_response: str
    user_question: str
    is_positive: bool
    conversation_id: str

# ================== Helper Functions ==================

def detect_user_lang(text: str) -> str:
    """Detect user language."""
    try:
        lang = detect((text or "").strip())
        return "en" if lang and lang.startswith("en") else "el"
    except LangDetectException:
        return "el"

def is_greeting(text: str) -> bool:
    """Check if text is a greeting."""
    greetings = ['γεία', 'γεια', 'καλημέρα', 'καλημερα', 'καλησπέρα', 'καλησπερα',
                 'χάιρετε', 'χαιρετε', 'hello', 'hi', 'good morning', 'good evening']
    text_lower = text.lower().strip()
    return any(text_lower.startswith(g) or text_lower == g for g in greetings)

def get_direct_answer(question: str) -> Optional[Dict]:
    """Return direct answers for common questions that semantic search might miss."""
    text_lower = question.lower().strip()
    
    # ΚΕΠ - Ώρες λειτουργίας
    if any(kw in text_lower for kw in ['κεπ', 'κέντρο εξυπηρέτησης πολιτών', 'center']):
        if any(kw in text_lower for kw in ['ώρες', 'ωράριο', 'λειτουργ', 'hours', 'time', 'when', 'πότε']):
            return {
                "answer": """ΚΕΠ Κορίνθου - Ωράριο Λειτουργίας

📍 Διεύθυνση: Κωστή Παλαμά 53, 20131 Κόρινθος

🕒 Ωράριο: Δευτέρα - Παρασκευή, 8:00 - 15:00
         (Τετάρτη επίσης 17:00 - 19:00)

📞 Τηλέφωνο: 2741363555
📧 Email: n.korinthias@kep.gov.gr""",
                "quality": "direct_match",
                "context_found": True,
                "confidence": 0.95
            }
        # ΚΕΠ - Στοιχεία επικοινωνίας
        if any(kw in text_lower for kw in ['τηλέφωνο', 'τηλ', 'email', 'address', 'διεύθυνση', 'επικοινωνία', 'στοιχεία']):
            return {
                "answer": """ΚΕΠ Κορίνθου - Στοιχεία Επικοινωνίας

📍 Διεύθυνση: Κωστή Παλαμά 53, 20131 Κόρινθος

📞 Τηλέφωνο: 2741363555
📧 Email: n.korinthias@kep.gov.gr

🕒 Ωράριο: Δευτέρα - Παρασκευή 8:00-15:00""",
                "quality": "direct_match",
                "context_found": True,
                "confidence": 0.95
            }
    
    # ΔΕΥΑ - Multiple variations
    if any(kw in text_lower for kw in ['δευα', 'δ.ε.υ.α', 'water', 'νερό']):
        if any(kw in text_lower for kw in ['τηλέφωνο', 'τηλ', 'κλήση', 'επικοινωνία', 'call', 'phone']):
            return {
                "answer": """ΔΕΥΑ Κορίνθου
📞 Τηλέφωνο Κέντρο: 2741024444
📞 Βλάβες (24ωρο): 6936776041
📧 Email: info@deyakor.gr""",
                "quality": "direct_match",
                "context_found": True,
                "confidence": 0.95
            }
    
    # Δήμαρχος - Multiple variations
    if any(kw in text_lower for kw in ['δήμαρχ', 'δημαρχ', 'mayor', 'αρχηγ']):
        if any(kw in text_lower for kw in ['τηλέφωνο', 'τηλ', 'email', 'επικοινωνία', 'στοιχεία', 'contact']):
            return {
                "answer": """Γραφείο Δημάρχου Κορινθίων

Δήμαρχος: Νίκος Σταυρέλης

📞 Τηλέφωνο: 27413-61001, 27413-61041
📧 Email: grafeiodimarxou@korinthos.gr

📍 Διεύθυνση: Κολιάτσου 32, 201 31 Κόρινθος""",
                "quality": "direct_match",
                "context_found": True,
                "confidence": 0.95
            }
    
    # Τηλέφωνα Δήμου - Multiple variations
    if any(kw in text_lower for kw in ['τηλέφωνα δήμου', 'δημοσιο τηλ', 'δημος κορινθ', 'κλήση δήμου', 'phone municipality']):
        if any(kw in text_lower for kw in ['τηλ', 'κέντρο', 'κλήση', 'phone', 'call', 'contact']):
            return {
                "answer": """Τηλεφωνικό Κέντρο Δήμου Κορινθίων
📞 27413-61000 (Κύρια γραμμή)
📞 27413-61040 (Εναλλακτική)
📞 27413-61045 (Γραφείο Τύπου)

Ωράριο: Δευτέρα-Παρασκευή 8:00-14:00

Για αιτήματα: protokollo@korinthos.gr""",
                "quality": "direct_match",
                "context_found": True,
                "confidence": 0.95
            }
    
    return None
    """Check if user is asking about bot's capabilities."""
    text_lower = text.lower().strip()
    capability_keywords = [
        'τι πληροφορίες', 'ποιες πληροφορίες', 'τι μπορεις', 'τι δυνατοτητες',
        'μπορεις να κανεις', 'μπορεις να μου παρεχεις', 'βοηθα', 'βοηθησεις',
        'what can you help', 'what information', 'what capabilities', 'can you help'
    ]
    return any(kw in text_lower for kw in capability_keywords)

def get_capabilities_response(lang: str = "el") -> str:
    """Return bot capabilities response - compact version with titles only."""
    if lang == "en":
        return """As a digital assistant for the Municipality of Corinth, I can help with:

1) Municipal Phone Numbers
2) Certificate Issuance
3) Registry Acts
4) Municipal History
5) General Municipal Services

For more information, visit https://korinthos.gr/"""
    else:
        return """Ως ψηφιακός βοηθός του Δήμου Κορινθίων, μπορώ να σας δώσω πληροφορίες για:

1) Τηλέφωνα του Δήμου
2) Έκδοση Πιστοποιητικών
3) Ληξιαρχικές Πράξεις
4) Ιστορία του Δήμου
5) Δημοτικές Υπηρεσίες

Για περισσότερες πληροφορίες, επισκεφθείτε https://korinthos.gr/"""

def is_capabilities_question(text: str) -> bool:
    """Check if user is asking about bot's capabilities."""
    text_lower = text.lower().strip()
    capability_keywords = [
        'τι πληροφορίες', 'ποιες πληροφορίες', 'τι μπορεις', 'τι δυνατοτητες',
        'μπορεις να κανεις', 'μπορεις να μου παρεχεις', 'βοηθα', 'βοηθησεις',
        'what can you help', 'what information', 'what capabilities', 'can you help',
        'τι ξερεις', 'τι γνωρίζεις', 'για τι', 'σχετικα με τι'
    ]
    return any(kw in text_lower for kw in capability_keywords)

def is_out_of_scope(text: str) -> bool:
    """Check if question is out of scope. Be lenient - let RAG decide."""
    text_lower = text.lower().strip()
    
    # Scope keywords - if ANY of these are present, it's IN scope
    scope_keywords = [
        'δήμος', 'κορίνθ', 'τηλέφωνο', 'επικοινωνία', 'πιστοποιητικό',
        'γέννησης', 'θανάτου', 'γάμου', 'ληξιαρχικές', 'μεταδημότευση',
        'υπηρεσία', 'δημαρχ', 'αίτηση', 'έγγραφο', 'ωράριο', 'gov.gr',
        'εκδοση', 'πρακτικα', 'διαδικασία', 'προϋποθέσεις', 'βήματα',
        'municipality', 'corinth', 'certificate', 'service', 'process'
    ]
    
    # If ANY keyword is present, it's likely IN scope - let RAG decide
    if any(kw in text_lower for kw in scope_keywords):
        return False  # IN SCOPE - let RAG handle it
    
    # Very obviously out of scope
    clearly_out_of_scope = [
        'weather', 'καιρός', 'football', 'ποδόσφαιρο', 'recipe', 'συνταγή',
        'movie', 'ταινία', 'politics', 'πολιτική', 'celebrity', 'σελέμπριτι'
    ]
    
    if any(kw in text_lower for kw in clearly_out_of_scope):
        return True  # OUT OF SCOPE
    
    # When in doubt, let RAG try - it's better to attempt than reject
    return False

def semantic_search(cursor, question: str, top_k: int = 5) -> List[Dict]:
    """
    Semantic search using embeddings.
    Returns top-k most relevant documents.
    """
    try:
        q_embedding = get_embedder().encode(question)
        q_embedding_list = q_embedding.tolist()
        
        # Get MORE results initially to have better selection
        cursor.execute("""
            SELECT id, question, answer, 
                   1 - (embedding_384 <=> %s::vector) as similarity
            FROM public.kb_items_raw 
            WHERE embedding_384 IS NOT NULL
            ORDER BY embedding_384 <-> %s::vector
            LIMIT %s
        """, (q_embedding_list, q_embedding_list, top_k * 3))  # Get 3x more initially
        
        results = []
        for r_id, r_question, r_answer, similarity in cursor.fetchall():
            if similarity > 0.0:  # Accept ANY similarity (will filter later)
                results.append({
                    "id": r_id,
                    "question": r_question,
                    "answer": r_answer,
                    "similarity": float(similarity),
                    "source": "semantic"
                })
        
        # Return top-k sorted by similarity
        results = sorted(results, key=lambda x: x['similarity'], reverse=True)[:top_k]
        
        top_sims = [f"{r['similarity']:.3f}" for r in results[:3]]
        log.info(f"🔍 Semantic search: {len(results)} results (top similarities: {top_sims})")
        return results
    
    except Exception as e:
        log.error(f"❌ Semantic search error: {e}")
        return []

def keyword_search(cursor, question: str, top_k: int = 3) -> List[Dict]:
    """
    Keyword search for exact matches.
    Useful when semantic search might miss specific terms.
    """
    try:
        q_lower = question.lower().strip()
        # Remove punctuation
        q_lower = q_lower.replace('?', '').replace(';', '').replace('!', '').replace(',', '')
        keywords = [kw.strip() for kw in q_lower.split() if len(kw.strip()) > 2]
        
        if not keywords:
            return []
        
        # Build OR conditions for all keywords
        conditions = " OR ".join([f"(LOWER(question) ILIKE %s OR LOWER(answer) ILIKE %s)" for _ in keywords])
        params = []
        for kw in keywords:
            params.extend([f"%{kw}%", f"%{kw}%"])
        params.append(top_k * 2)  # Get more results
        
        cursor.execute(f"""
            SELECT id, question, answer, 0.95 as similarity
            FROM public.kb_items_raw 
            WHERE {conditions}
            ORDER BY id
            LIMIT %s
        """, params)
        
        results = []
        seen_ids = set()
        for r_id, r_question, r_answer, similarity in cursor.fetchall():
            if r_id not in seen_ids:
                results.append({
                    "id": r_id,
                    "question": r_question,
                    "answer": r_answer,
                    "similarity": float(similarity),
                    "source": "keyword"
                })
                seen_ids.add(r_id)
        
        log.info(f"🔑 Keyword search: {len(results)} results (keywords: {keywords})")
        return results
    
    except Exception as e:
        log.error(f"❌ Keyword search error: {e}")
        return []

def retrieve_context(cursor, question: str, top_k: int = 5) -> List[Dict]:
    """
    RAG Step 1: Optimized RETRIEVE
    Μειώνουμε το top_k για ταχύτητα. Το GPT-4o-mini αποδίδει καλύτερα 
    με λιγότερο και πιο σχετικό περιεχόμενο.
    """
    # 1. Semantic Search (top_k=5 αντί για 8)
    semantic_results = semantic_search(cursor, question, top_k=5)
    
    # 2. Keyword Search (top_k=2 αντί για 4)
    keyword_results = keyword_search(cursor, question, top_k=2)

    all_results = {}
    
    # Προσθήκη keyword results
    for doc in keyword_results:
        all_results[doc['id']] = doc
    
    # Προσθήκη semantic results & Deduplication
    for doc in semantic_results:
        if doc['id'] not in all_results:
            all_results[doc['id']] = doc
        else:
            all_results[doc['id']]['similarity'] = max(doc.get('similarity', 0), all_results[doc['id']].get('similarity', 0))
    
    # Επιστρέφουμε μόνο τα 5 κορυφαία σε βαθμολογία
    return sorted(all_results.values(), key=lambda x: x.get('similarity', 0), reverse=True)[:5]
    
    # Καταγραφή πληροφοριών στο Log για έλεγχο στο τερματικό σου
    log.info(f"📂 Hybrid Search: Top Semantic Sim: {top_sim:.3f}")
    log.info(f"📚 Total unique documents collected for GPT: {len(all_results)}")

    # 3. Ταξινόμηση βάσει συνάφειας και επιστροφή
    # Χρησιμοποιούμε το top_k που έχει οριστεί (συνήθως 10-15 για το GPT)
    ranked = sorted(all_results.values(), key=lambda x: x.get('similarity', 0), reverse=True)[:top_k]
    
    # ΕΠΙΣΤΡΟΦΗ ΤΗΣ ΛΙΣΤΑΣ ΣΤΗ ΣΥΝΑΡΤΗΣΗ generate_answer_with_ragσ
    return ranked

def format_context(docs: List[Dict], lang: str = "el") -> str:
    """Format retrieved documents for the LLM prompt."""
    if not docs:
        if lang == "en":
            return "(No relevant documents found in the knowledge base)"
        else:
            return "(Δεν βρέθηκαν σχετικά έγγραφα στη βάση γνώσης)"
    
    if lang == "en":
        formatted = "Knowledge Base Documents:\n" + "="*60 + "\n"
    else:
        formatted = "Έγγραφα από τη Βάση Γνώσης:\n" + "="*60 + "\n"
    
    for i, doc in enumerate(docs, 1):
        formatted += f"\n[Document {i}] (Relevance: {doc['similarity']:.0%})\n"
        formatted += f"Q: {doc['question']}\n"
        formatted += f"A: {doc['answer']}\n"
    
    return formatted

async def generate_answer_with_rag(question: str, context_str: str, 
                                   lang: str = "el", conversation_history: List[Dict] = None) -> Tuple[str, Dict]:
    """
    Παραγωγή AI απάντησης με πλήρη αξιοποίηση του GPT-4o-mini και των 88 πληροφοριών.
    """
    # ΔΙΟΡΘΩΣΗ: Χρησιμοποιούμε το context_str που έρχεται ως όρισμα
    log.info(f"🤖 Generating AI response in '{lang}'...")
    
    # Αν το context_str είναι κενό, βάζουμε ένα default μήνυμα
    if not context_str or not context_str.strip():
        current_context = "Δεν βρέθηκαν σχετικά έγγραφα στη βάση δεδομένων."
    else:
        current_context = context_str

    # 2. Το "Ελεύθερο" αλλά "Πειθαρχημένο" System Prompt
    if lang == "el":
        system_prompt = (
         "Είσαι η Εφύρα, η προηγμένη AI βοηθός του Δήμου Κορινθίων. Η αποστολή σου είναι να εξυπηρετείς τους πολίτες "
    "με φυσικό, ανθρώπινο και επαγγελματικό τρόπο, χρησιμοποιώντας αποκλειστικά την επίσημη βάση γνώσης του Δήμου.\n\n"
    
    # Η ΚΡΙΣΙΜΗ ΠΡΟΣΘΗΚΗ ΓΙΑ ΤΗ ΓΛΩΣΣΑ:
    "ΓΛΩΣΣΙΚΟΣ ΚΑΝΟΝΑΣ: Απάντησε ΠΑΝΤΑ στη γλώσσα που σου απευθύνεται ο χρήστης. "
    "Αν ο χρήστης ρωτήσει στα Αγγλικά, μετέφρασε τις πληροφορίες από το ελληνικό CONTEXT και απάντησε στα Αγγλικά. "
    "Αν ρωτήσει στα Ελληνικά, απάντησε στα Ελληνικά.\n\n"

    "ΚΑΝΟΝΕΣ ΛΕΙΤΟΥΡΓΙΑΣ:\n"
    "1. ΠΡΟΤΕΡΑΙΟΤΗΤΑ ΔΕΔΟΜΕΝΩΝ: Χρησιμοποίησε ΜΟΝΟ τις πληροφορίες που παρέχονται στο CONTEXT. "
    "Αγνόησε οποιαδήποτε προϋπάρχουσα γνώση από την εκπαίδευσή σου που έρχεται σε σύγκρουση (π.χ. παλιούς δημάρχους). "
    "Για εσένα, Δήμαρχος είναι ο ΝΙΚΟΣ ΣΤΑΥΡΕΛΗΣ.\n"
    
    "2. ΑΚΡΙΒΕΙΑ ΤΗΛΕΦΩΝΩΝ (ΣΚΟΝΑΚΙ): Για τις παρακάτω υπηρεσίες, χρησιμοποίησε ΑΥΣΤΗΡΑ αυτά τα νούμερα:\n"
            "   - Βλάβες Ηλεκτροφωτισμού: 2741120134\n"
            "   - Βλάβες ΔΕΥΑ (Νερό): 2741024444 (24ωρο: 6936776041)\n"
            "   - Γραφείο Δημάρχου: 2741361041\n"
            "   - Τηλεφωνικό Κέντρο: 2741361000\n"
            "   Αν ο χρήστης ρωτήσει για κάτι άλλο που δεν υπάρχει στο Context, δώσε το γενικό 2741361000.\n"
    
    "3. ΦΥΣΙΚΟΣ ΛΟΓΟΣ (AI-Powered): Μην αναφέρεις ΠΟΤΕ τις λέξεις 'Context', 'έγγραφα' ή 'βάση δεδομένων'. "
    "Μην λες 'Σύμφωνα με το έγγραφο 1'. Απάντησε απευθείας: 'Με βάση την ενημέρωση του Δήμου...' ή 'Μπορείτε να καλέσετε στο...'. "
    "Σύνθεσε μια απάντηση που ρέει φυσικά, συνδυάζοντας πληροφορίες αν χρειαστεί.\n"
    
    "4. ΔΙΑΧΕΙΡΙΣΗ ΠΑΡΑΛΛΑΓΩΝ: Κατανόησε το νόημα της ερώτησης. Αν κάποιος ρωτήσει 'ποιος κάνει κουμάντο' "
    "ή 'ποιος είναι ο αρχηγός', κατάλαβε ότι εννοεί τον Δήμαρχο.\n"
    
    "5. ΑΓΝΩΣΤΗ ΠΛΗΡΟΦΟΡΙΑ: Αν η πληροφορία δεν υπάρχει καθόλου στις 88 εγγραφές, μην μαντέψεις. "
    "Απάντησε ευγενικά ότι δεν διαθέτεις τη συγκεκριμένη πληροφορία και παρέπεμψε στο korinthos.gr ή στο 2741361000."
)
    else:
        system_prompt = (
            "You are Ephyra, the advanced AI assistant for the Municipality of Corinth. "
            "Your mission is to serve citizens in a natural, human, and professional manner, "
            "using EXCLUSIVELY the official knowledge base of the Municipality.\n\n"
            
            "OPERATIONAL RULES:\n"
            "1. DATA PRIORITY: Use ONLY the information provided in the CONTEXT. "
            "Ignore any prior knowledge from your training that conflicts with this data (e.g., former mayors). "
            "For you, the Mayor is NIKOS STAVRELIS.\n"
            
            "2. PHONE ACCURACY: Never invent phone numbers. If the user asks about faults or services, "
            "provide the exact number mentioned in the corresponding document (e.g., for street lighting use 2741120134). "
            "If no specific number is found, use the general center 2741361000.\n"
            
            "3. NATURAL LANGUAGE (AI-Powered): NEVER mention the words 'Context', 'documents', or 'database'. "
            "Do not say 'According to document 1'. Answer directly: 'Based on the Municipality's information...' "
            "or 'You can call...'. Compose a response that flows naturally, combining information if necessary.\n"
            
            "4. LANGUAGE RULE: Always answer in the language the user addresses you in. "
            "Since the CONTEXT is in Greek, you must accurately translate the information into English for the user.\n"
            
            "5. HANDLING VARIATIONS: Understand the meaning of the question. If someone asks 'who is in charge' "
            "or 'who is the boss', understand they mean the Mayor.\n"
            
            "6. UNKNOWN INFORMATION: If the information is not in the 88 records, do not guess. "
            "Politely state that you do not have this information and refer them to korinthos.gr or 2741361000."
        )
        

    try:
        # Κλήση OpenAI
        response = openai_client.chat.completions.create(
            model="gpt-4o-mini",
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "system", "content": f"CONTEXT:\n{current_context}"},
                {"role": "user", "content": question}
            ],
            temperature=0.7
        )
        answer = response.choices[0].message.content.strip()
        
        # Απλοποιημένα Metadata για να μην έχουμε σφάλματα 'get'
        metadata = {
            "documents_used": 1 if context_str.strip() else 0,
            "source": "hybrid_knowledge_base"
        }
        return answer, metadata

    except Exception as e:
        log.error(f"❌ OpenAI Error: {e}")
        return "Λυπάμαι, δεν μπόρεσα να επεξεργαστώ την απάντηση.", {}

# ================== Endpoints ==================

@app.get("/")
async def root(background_tasks: BackgroundTasks):
    # Αυτό λέει στην Python: "Δείξε το site αμέσως και ξεκίνα τον συγχρονισμό από πίσω"
    background_tasks.add_task(sync_csv_to_db) 
    
    current_dir = os.path.dirname(os.path.abspath(__file__))
    html_path = os.path.join(current_dir, "ui_chatbot.html")
    
    if os.path.exists(html_path):
        return FileResponse(html_path, media_type="text/html")
    return {"message": "Ephyra is warming up!"}

@app.get("/health")
async def health():
    """Health check endpoint."""
    return {
        "status": "healthy",
        "version": "3.0.0-RAG",
        "architecture": "Retrieval Augmented Generation with OpenAI GPT-4o-mini"
    }

@app.get("/dashboard")
async def get_dashboard():
    """Endpoint για την προβολή του Feedback Dashboard."""
    current_dir = os.path.dirname(os.path.abspath(__file__))
    dashboard_path = os.path.join(current_dir, "feedback_dashboard.html")
    
    if os.path.exists(dashboard_path):
        return FileResponse(dashboard_path, media_type="text/html")
    return {"error": "Dashboard file not found"}

@app.get("/questionnaire")
async def get_questionnaire():
    """Endpoint για την προβολή του Ερωτηματολογίου."""
    current_dir = os.path.dirname(os.path.abspath(__file__))
    # Σιγουρέψου ότι το όνομα του αρχείου είναι ακριβώς questionnaire.html
    quest_path = os.path.join(current_dir, "questionnaire.html")
    
    if os.path.exists(quest_path):
        return FileResponse(quest_path, media_type="text/html")
    return {"error": "Questionnaire file not found"}

@app.post("/ask")
@limiter.limit("30/minute")
async def ask(request: Request, body: AskBody):
    # 1. Αρχικοποίηση
    current_lang = body.lang or "el"
    question = (body.messages[-1].content if body.messages else "").strip()

    if not question:
        return {"answer": "Δεν έλαβα ερώτηση", "quality": "error"}

    # --- ΝΕΟ ΚΟΜΜΑΤΙ: Έλεγχος για άμεση απάντηση (Cheat Sheet) ---
    direct_resp = get_direct_answer(question)
    if direct_resp:
        # Αν βρούμε άμεση απάντηση, τη στέλνουμε αμέσως σαν stream!
        async def direct_stream():
            yield direct_resp["answer"]
        return StreamingResponse(direct_stream(), media_type="text/plain")
    # -------------------------------------------------------------

    # 2. Προετοιμασία Context από CSV (αν δεν υπάρχει άμεση απάντηση)
    csv_context = ""
    
    def clean_text(t):
        if not t: return ""
        return t.lower().translate(str.maketrans('', '', string.punctuation)).strip()

    clean_user_q = clean_text(question)

    # Σάρωση του CSV για γρήγορες απαντήσεις
    for row in knowledge_base:
        values = list(row.values())
        if len(values) >= 2:
            csv_q_raw = str(values[0])
            csv_a = values[1]
            
            clean_csv_q = clean_text(csv_q_raw)
            if clean_csv_q and (clean_csv_q in clean_user_q or clean_user_q in clean_csv_q):
                csv_context += f"\nΣχετική πληροφορία από CSV: {csv_a}\n"

    async def event_generator():
        conn = get_db_conn()
        try:
            cursor = conn.cursor()
            
            # 3. Λήψη Context από τη Βάση (Retrieve)
            db_context_docs = retrieve_context(cursor, question, top_k=5)
            db_context_text = ""
            for doc in db_context_docs:
                q = doc.get('question', '')
                a = doc.get('answer', '')
                db_context_text += f"\nΠληροφορία: {q} - {a}\n"
            cursor.close()

            all_context = csv_context + "\n" + db_context_text

            # 4. Κλήση OpenAI
            response = openai_client.chat.completions.create(
                model="gpt-4o-mini",
                messages=[
                    {"role": "system", "content": f"Είσαι η Εφύρα, η ψηφιακή βοηθός του Δήμου Κορινθίων. Απάντησε στη γλώσσα: {current_lang}. Χρησιμοποίησε το παρακάτω CONTEXT για να απαντήσεις."},
                    {"role": "system", "content": f"CONTEXT:\n{all_context}"},
                    {"role": "user", "content": question}
                ],
                temperature=0.7,
                stream=True
            )

            for chunk in response:
                if chunk.choices and chunk.choices[0].delta.content:
                    content = chunk.choices[0].delta.content
                    yield content

        except Exception as e:
            log.error(f"❌ Streaming Error: {e}")
            yield " Λυπάμαι, παρουσιάστηκε ένα πρόβλημα στη σύνδεση."
        finally:
            if conn:
                return_db_conn(conn)

    return StreamingResponse(event_generator(), media_type="text/plain")
    # ----------------------------------------------
     

@app.post("/feedback")
@limiter.limit("10/minute")
async def record_feedback(request: Request):
    """Record user feedback."""
    conn = None
    try:
        data = await request.json()
        conn = get_db_conn()
        cursor = conn.cursor()
        
        conversation_id = data.get("conversation_id", str(uuid.uuid4()))
        bot_response = data.get("bot_response", "")
        user_question = data.get("user_question", "")
        is_positive = data.get("is_positive")
        
        if is_positive is None:
            return {"status": "error", "message": "is_positive field required"}
        
        user_agent = request.headers.get("User-Agent", "Unknown")
        client_ip = request.client.host if request.client else "Unknown"
        
        cursor.execute("""
            INSERT INTO chatbot_feedback 
            (conversation_id, bot_response, user_question, is_positive, user_agent, ip_address)
            VALUES (%s, %s, %s, %s, %s, %s)
        """, (
            conversation_id,
            bot_response[:5000],
            user_question[:2000] if user_question else None,
            is_positive,
            user_agent[:255],
            client_ip
        ))
        
        conn.commit()
        feedback_type = "✅ Positive" if is_positive else "❌ Negative"
        log.info(f"📊 Feedback recorded: {feedback_type}")
        
        return {
            "status": "success",
            "message": "Thank you for your feedback!",
            "feedback_id": conversation_id
        }
    
    except Exception as e:
        log.exception(f"❌ Feedback error: {e}")
        if conn:
            conn.rollback()
        return {"status": "error", "message": str(e)}
    
    finally:
        if conn:
            return_db_conn(conn)

@app.get("/tts_play")
@limiter.limit("15/minute")
async def tts_play(request: Request, text: str = ""):
    """Text-to-Speech endpoint."""
    text = (text or "").strip()
    if not text:
        raise HTTPException(status_code=400, detail="Missing text parameter")
    if len(text) > 5000:
        raise HTTPException(status_code=400, detail="Text too long for TTS")
    
    audio_data = await _elevenlabs_tts_request(text)
    return StreamingResponse(io.BytesIO(audio_data), media_type="audio/mpeg")

@app.post("/tts_play")
@limiter.limit("15/minute")
async def tts_play_post(request: Request, body: TTSBody):
    """Text-to-Speech endpoint (POST)."""
    text = (body.text or "").strip()
    if not text:
        raise HTTPException(status_code=400, detail="Missing text")
    if len(text) > 5000:
        raise HTTPException(status_code=400, detail="Text too long for TTS")
    
    audio_data = await _elevenlabs_tts_request(text)
    return StreamingResponse(io.BytesIO(audio_data), media_type="audio/mpeg")



async def _elevenlabs_tts_request(text: str) -> bytes:
    if not ELEVENLABS_API_KEY:
        raise HTTPException(status_code=400, detail="API Key missing")
    
    try:
        # 1. Καθαρισμός από Emojis (παραμένει όπως πριν)
        clean_text = text.replace("📍", "").replace("🕒", "").replace("📞", "").replace("📧", "").strip()
        
        # 2. Διαχωρισμός τηλεφώνων (π.χ. 2741024444 -> 2 7 4 1 0 2 4 4 4 4)
        # Αυτό αναγκάζει την ElevenLabs να διαβάζει ψηφίο-ψηφίο
        clean_text = re.sub(r'(\d)', r'\1 ', clean_text)
        
        # 3. Ειδική διόρθωση για το "24ωρο" ή "8:00" αν χρειάζεται
        clean_text = clean_text.replace(" : ", ":") # Επαναφορά ώρας αν χάλασε από τα κενά
        
        audio_stream = eleven_client.text_to_speech.convert(
            voice_id=ELEVENLABS_VOICE_ID,
            text=clean_text,
            model_id="eleven_multilingual_v2",
            output_format="mp3_44100_128"
        )
        
        audio_data = b""
        for chunk in audio_stream:
            if chunk: audio_data += chunk
        return audio_data

    except Exception as e:
        log.error(f"❌ ElevenLabs Error: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))
    
    
@app.on_event("shutdown")
def shutdown_event():
    """Cleanup on shutdown."""
    try:
        conn_pool.closeall()
        log.info("🔌 Database connection pool closed")
    except Exception:
        pass

# ================== FEEDBACK ENDPOINTS ==================

@app.get("/feedback/stats")
async def get_feedback_stats(days: int = 30, detailed: bool = False):
    """Get feedback statistics with advanced metrics."""
    conn = None  # Αρχικοποίηση εκτός του try
    try:
        conn = get_db_conn()
        cursor = conn.cursor()
        
        since_date = datetime.now() - timedelta(days=days)
        
        # Basic stats
        cursor.execute("""
            SELECT 
                COUNT(*) as total,
                SUM(CASE WHEN is_positive THEN 1 ELSE 0 END) as positive,
                SUM(CASE WHEN is_positive THEN 0 ELSE 1 END) as negative
            FROM chatbot_feedback
            WHERE timestamp >= %s
        """, (since_date,))
        
        result = cursor.fetchone()
        total = result[0] or 0 if result else 0
        positive = result[1] or 0 if result else 0
        negative = result[2] or 0 if result else 0
        
        satisfaction_rate = round((positive / total * 100)) if total > 0 else 0
        
        # Daily data for chart
        cursor.execute("""
            SELECT 
                DATE(timestamp) as date,
                is_positive,
                COUNT(*) as count
            FROM chatbot_feedback
            WHERE timestamp >= %s
            GROUP BY DATE(timestamp), is_positive
            ORDER BY date
        """, (since_date,))
        
        daily_data = []
        for row in cursor.fetchall():
            if row:
                daily_data.append({
                    "date": str(row[0]) if row[0] else "",
                    "sentiment": "positive" if row[1] else "negative",
                    "count": row[2] or 0
                })
        
        # Top questions
        cursor.execute("""
            SELECT user_question, COUNT(*) as count
            FROM chatbot_feedback
            WHERE timestamp >= %s AND user_question IS NOT NULL AND user_question != ''
            GROUP BY user_question
            ORDER BY count DESC
            LIMIT 5
        """, (since_date,))
        
        top_questions = [{"question": row[0] or "", "count": row[1] or 0} for row in cursor.fetchall()]
        
        # Language distribution
        cursor.execute("""
            SELECT 
                CASE 
                    WHEN user_question ~ '[α-ωΑ-Ωά-ώΆ-Ώ]' THEN 'el' 
                    ELSE 'en' 
                END as lang,
                COUNT(*) as count
            FROM chatbot_feedback
            WHERE timestamp >= %s
            GROUP BY lang
        """, (since_date,))
        
        language_distribution = {"el": 0, "en": 0}
        
        for row in cursor.fetchall():
            lang_code = row[0]
            count = row[1]
            if lang_code in language_distribution:
                language_distribution[lang_code] = count
        
        
        # Sentiment trend
        prev_since = since_date - timedelta(days=days)
        cursor.execute("""
            SELECT 
                SUM(CASE WHEN is_positive THEN 1 ELSE 0 END) as positive,
                COUNT(*) as total
            FROM chatbot_feedback
            WHERE timestamp >= %s AND timestamp < %s
        """, (prev_since, since_date))
        
        prev_result = cursor.fetchone()
        prev_satisfaction = 0
        if prev_result and prev_result[1] and prev_result[1] > 0:
            prev_satisfaction = round((prev_result[0] / prev_result[1] * 100))
        
        sentiment_trend = satisfaction_rate - prev_satisfaction
        
        # User metrics
        cursor.execute("""
            SELECT COUNT(DISTINCT ip_address) as unique_users
            FROM chatbot_feedback
            WHERE timestamp >= %s
        """, (since_date,))
        
        user_result = cursor.fetchone()
        unique_users = user_result[0] if user_result and user_result[0] else 0
        
        # Recent feedback
        recent_feedback = []
        if detailed:
            cursor.execute("""
                SELECT id, user_question, bot_response, is_positive, timestamp, ip_address
                FROM chatbot_feedback
                WHERE timestamp >= %s
                ORDER BY timestamp DESC
                LIMIT 50
            """, (since_date,))
            
            for row in cursor.fetchall():
                recent_feedback.append({
                    "id": row[0],
                    "user_question": row[1] or "",
                    "bot_response": row[2] or "",
                    "is_positive": row[3],
                    "timestamp": row[4].isoformat() if row[4] else None,
                    "ip_address": row[5],
                    "language": "el",
                    "response_time": 1.0
                })
        
        cursor.close()
        # ΠΡΟΣΟΧΗ: Εδώ σβήσαμε το conn.close() που προκαλούσε το πρόβλημα
        
        return {
            "total_feedback": total,
            "positive": positive,
            "negative": negative,
            "satisfaction_rate": satisfaction_rate,
            "daily_data": daily_data,
            "top_issues": [],
            "top_questions": top_questions,
            "language_distribution": language_distribution,
            "sentiment_trend": sentiment_trend,
            "sentiment_trend_percent": f"+{sentiment_trend}%" if sentiment_trend > 0 else f"{sentiment_trend}%",
            "avg_response_time": 1.2,
            "min_response_time": 0.5,
            "max_response_time": 3.0,
            "unique_users": unique_users,
            "weekly_active_users": unique_users // 2,
            "recent_feedback": recent_feedback
        }
        
    except Exception as e:
        log.error(f"Error getting feedback stats: {e}")
        return {"error": str(e)}
    finally:
        # ΑΥΤΗ ΕΙΝΑΙ Η ΔΙΟΡΘΩΣΗ: Επιστροφή στο Pool ό,τι και να γίνει
        if conn:
            return_db_conn(conn)


@app.get("/feedback/export")
async def export_feedback(days: int = 30, format: str = "csv"):
    """Export feedback data."""
    try:
        conn = get_db_conn()
        cursor = conn.cursor()
        
        since_date = datetime.now() - timedelta(days=days)
        
        cursor.execute("""
            SELECT id, user_question, bot_response, is_positive, timestamp, ip_address
            FROM chatbot_feedback
            WHERE timestamp >= %s
            ORDER BY timestamp DESC
        """, (since_date,))
        
        rows = cursor.fetchall()
        cursor.close()
        conn.close()
        
        if format == "csv":
            import io
            import csv
            
            output = io.StringIO()
            writer = csv.writer(output)
            writer.writerow(['ID', 'Ερώτηση', 'Απάντηση', 'Sentiment', 'Ημερομηνία', 'IP'])
            
            for row in rows:
                writer.writerow([
                    row[0],
                    row[1] or '',
                    row[2] or '',
                    '👍 Θετικό' if row[3] else '👎 Αρνητικό',
                    row[4].strftime('%Y-%m-%d %H:%M:%S') if row[4] else '',
                    row[5]
                ])
            
            return StreamingResponse(
                iter([output.getvalue()]),
                media_type="text/csv",
                headers={"Content-Disposition": "attachment; filename=feedback_export.csv"}
            )
        
    except Exception as e:
        log.error(f"Error exporting feedback: {e}")
        return {"error": str(e)}


@app.post("/feedback/clear")
async def clear_all_feedback():
    """Delete all feedback data."""
    try:
        conn = get_db_conn()
        cursor = conn.cursor()
        
        # Delete all feedback
        cursor.execute("DELETE FROM chatbot_feedback")
        conn.commit()
        
        log.warning("🗑️ ALL FEEDBACK DELETED")
        
        return {
            "status": "success",
            "message": "✅ Όλα τα feedback διαγράφηκαν επιτυχώς!"
        }
        
    except Exception as e:
        log.error(f"Error clearing feedback: {e}")
        if conn:
            conn.rollback()
        return {"status": "error", "message": str(e)}
    
    finally:
        if conn:
            return_db_conn(conn)


from pydantic import BaseModel

# --- SURVEY SYSTEM (START) ---



# 2. Το "μονοπάτι" για την αποθήκευση στη βάση
@app.post("/submit_survey")
async def submit_survey(data: SurveyResponse):
    conn = get_db_conn()
    cur = conn.cursor()
    try:
        # 21 πεδία συνολικά (μαζί με gender, age, q16)
        query = """
            INSERT INTO survey_final 
            (used_bot, usage_context, scenarios_tested, gender, age, 
             q1, q2, q3, q4, q5, q6, q7, q8, q9, q10, 
             q11, q12, q13, q14, q15, q16, comments)
            VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s)
        """
        cur.execute(query, (
            data.usedBot, data.usageContext, data.scenarios, data.gender, data.age,
            data.q1, data.q2, data.q3, data.q4, data.q5,
            data.q6, data.q7, data.q8, data.q9, data.q10,
            data.q11, data.q12, data.q13, data.q14, data.q15, data.q16,
            data.comments
        ))
        conn.commit()
        return {"status": "success"}
    except Exception as e:
        logging.error(f"Survey Error: {e}")
        return {"status": "error", "message": str(e)}
    finally:
        cur.close()
        return_db_conn(conn)

# 3. Το "μονοπάτι" για να βλέπουμε τα αποτελέσματα στο Dashboard
@app.get("/survey_results")
async def get_survey_final():
    conn = None
    try:
        conn = get_db_conn()
        cur = conn.cursor()
        
        cur.execute("""
            SELECT id, timestamp, scenarios_tested, gender, age, 
                   q1, q2, q3, q4, q5, q6, q7, q8, q9, q10, 
                   q11, q12, q13, q14, q15, q16 
            FROM survey_final 
            ORDER BY timestamp DESC
        """)
        rows = cur.fetchall()
        
        results = []
        for r in rows:
            results.append({
                "id": r[0],
                "timestamp": r[1].strftime("%Y-%m-%d %H:%M:%S") if r[1] else "",
                "scenarios": r[2], # ΝΕΟ
                "gender": r[3],
                "age": r[4],
                "q1": r[5], "q2": r[6], "q3": r[7], "q4": r[8], "q5": r[9],
                "q6": r[10], "q7": r[11], "q8": r[12], "q9": r[13], "q10": r[14],
                "q11": r[15], "q12": r[16], "q13": r[17], "q14": r[18], "q15": r[19],
                "q16": r[20]
            })
        cur.close()
        return results
    except Exception as e:
        log.error(f"Error getting survey results: {e}")
        return []
    finally:
        if conn: return_db_conn(conn)

# ✅ 3. ΤΕΛΕΥΤΑΙΟ ΣΤΟ ΑΡΧΕΙΟ: Η ΕΚΚΙΝΗΣΗ
if __name__ == "__main__":
    import uvicorn
    log.info("🚀 Ephyra Chatbot v3.0.0 - Production RAG Edition starting...")
    uvicorn.run(app, host="0.0.0.0", port=8000)

# --- SURVEY SYSTEM (END) ---