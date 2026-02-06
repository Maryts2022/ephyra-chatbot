"""
Ephyra Chatbot - Production RAG
Final Version: Fixed Negative Stats Logic (Ignore Neutrals)
"""

import os
import io
import logging
import csv
import string
import re
from functools import lru_cache
from pathlib import Path
from typing import Optional, List, Dict
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
from langdetect import detect

# ================== 1. Configuration & Setup ==================

load_dotenv(dotenv_path=Path(__file__).with_name(".env"), override=True)

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
log = logging.getLogger("ephyra")

required_vars = ["OPENAI_API_KEY", "DB_NAME", "DB_USER", "DB_PASS", "DB_HOST"]
missing = [v for v in required_vars if not os.getenv(v)]
if missing:
    raise RuntimeError(f"Missing env vars: {', '.join(missing)}")

openai_client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))

ELEVENLABS_API_KEY = (os.getenv("ELEVENLABS_API_KEY") or "").strip()
ELEVENLABS_VOICE_ID = (os.getenv("ELEVENLABS_VOICE_ID") or "EXAVITQu4vr4xnSDxMaL").strip()
eleven_client = ElevenLabs(api_key=ELEVENLABS_API_KEY)

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

# ================== 3. Database Initialization ==================

def init_all_tables():
    conn = get_db_conn()
    cur = conn.cursor()
    try:
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
        cur.execute("""
            CREATE TABLE IF NOT EXISTS survey_final_v2 (
                id SERIAL PRIMARY KEY,
                timestamp TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                used_bot TEXT,
                user_status TEXT,
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
    except Exception as e:
        log.error(f"❌ Error initializing tables: {e}")
        if conn: conn.rollback()
    finally:
        cur.close()
        return_db_conn(conn)

# ================== 4. Background Tasks ==================

def sync_csv_to_db():
    try:
        conn = get_db_conn()
        cur = conn.cursor()
        log.info("🔄 Syncing CSV to DB...")
        
        # Χρήση του global embedder για ταχύτητα
        model = get_embedder()
        
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

# ================== 5. Helper Functions ==================

# Proactive Loading (για ταχύτητα)
log.info("🚀 Pre-loading AI Model into Memory...")
global_embedder = SentenceTransformer("paraphrase-multilingual-MiniLM-L12-v2")
log.info("✅ AI Model Loaded & Ready!")

def get_embedder():
    return global_embedder

# --- STATIC KNOWLEDGE ---
STATIC_KNOWLEDGE = """
[STANDARD MUNICIPAL INFO]

1. ΔΕΥΑ Κορίνθου (Ύδρευση/Αποχέτευση):
   - Τηλέφωνο Βλαβών (24ωρο): 6936776041
   - Τηλεφωνικό Κέντρο: 2741024444
   - Email: info@deyakor.gr

2. Ιστορία:
   - Η Κόρινθος καταστράφηκε από μεγάλους σεισμούς το 1858 και το 1928.
   - Μετά τον σεισμό του 1858, η πόλη μεταφέρθηκε στη σημερινή της θέση (Νέα Κόρινθος).

3. Τουρισμός - Αξιοθέατα (ΟΛΑ τα διαθέσιμα):
   - Αρχαία Κόρινθος & Αρχαιολογικό Μουσείο.
   - Ακροκόρινθος (Κάστρο).
   - Διώρυγα της Κορίνθου (Ισθμός).
   - Παραλία Καλάμια (Κέντρο πόλης).
   - Παραλία Λουτρά Ωραίας Ελένης.
   - Λαογραφικό Μουσείο Κορίνθου.
   - Ιερός Ναός Αποστόλου Παύλου.
"""

def get_direct_answer(question: str) -> Optional[Dict]:
    text_lower = question.lower().strip()
    
    # --- 1. GENERAL HELP ---
    if any(kw in text_lower for kw in ['τι μπορείς να κάνεις', 'τι ξέρεις', 'τι πληροφορίες', 'βοήθεια', 'δυνατότητες', 'help', 'what can you do']):
        return {
            "answer": """Μπορώ να σας δώσω πληροφορίες για:
✅ Διαδικασίες (Πιστοποιητικά, Ληξιαρχείο, Μεταδημότευση)
✅ Επικοινωνία (Τηλέφωνα, Email, Ωράρια Υπηρεσιών & ΚΕΠ)
✅ Τουρισμό (Αξιοθέατα, Μουσεία, Παραλίες)
✅ Διοίκηση (Δήμαρχος, Αντιδήμαρχοι)

Πληκτρολογήστε την ερώτησή σας!""",
            "quality": "direct_match"
        }

    # --- 2. DEPUTY MAYORS (ΠΡΟΤΕΡΑΙΟΤΗΤΑ) ---
    if 'deputy mayor' in text_lower or 'vice mayor' in text_lower or 'αντιδήμαρχ' in text_lower or 'αντιδημαρχ' in text_lower:
        
        if 'καθαριότ' in text_lower or 'καθαριοτ' in text_lower:
             return {"answer": "Αντιδήμαρχος Καθαριότητας: κ. Δημήτριος Μανωλάκης (Τηλ: 2741361000)", "quality": "direct_match"}
        if 'τουρισμ' in text_lower or 'παιδεία' in text_lower:
             return {"answer": "Αντιδήμαρχος Παιδείας & Τουρισμού: κ. Ευάγγελος Παπαϊωάννου", "quality": "direct_match"}
        if 'τεχνικ' in text_lower:
             return {"answer": "Αντιδήμαρχος Τεχνικών Υπηρεσιών: κ. Ανδρέας Ζώγκος", "quality": "direct_match"}
        if 'πολιτισμ' in text_lower:
             return {"answer": "Αντιδήμαρχος Πολιτισμού: κ. Αναστάσιος Ταγαράς", "quality": "direct_match"}
        if 'πολεοδομ' in text_lower:
             return {"answer": "Αντιδήμαρχος Πολεοδομίας: κ. Βασίλειος Πανταζής", "quality": "direct_match"}
        if 'διοικητ' in text_lower:
             return {"answer": "Αντιδήμαρχος Διοικητικών Υπηρεσιών: κ. Γεώργιος Πούρος", "quality": "direct_match"}

        return {
            "answer": """Οι Αντιδήμαρχοι είναι:
1. Γ. Πούρος (Διοικητικών)
2. Β. Πανταζής (Πολεοδομίας)
3. Δ. Μανωλάκης (Καθαριότητας)
4. Ε. Παπαϊωάννου (Παιδείας/Τουρισμού)
5. Α. Ζώγκος (Τεχνικών)
6. Α. Ταγαράς (Πολιτισμού)""",
            "quality": "direct_match"
        }
    
    # --- 3. MAYOR & LOCATION ---
    if any(kw in text_lower for kw in ['mayor', 'municipal', 'town hall', 'δημαρχείο', 'δήμαρχος', 'δημαρχο']):
        return {
            "answer": """Δημαρχείο Κορινθίων:
Δήμαρχος: **Νίκος Σταυρέλης**
📍 Διεύθυνση: Κολιάτσου 32, 201 31 Κόρινθος
📞 Τηλέφωνο: 27413-61001
📧 Email: grafeiodimarxou@korinthos.gr""", "quality": "direct_match"
        }

    return None

def retrieve_context(cursor, question: str, top_k: int = 5) -> List[Dict]:
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

# ================== 6. FastAPI App ==================

app = FastAPI(title="Ephyra Chatbot - Production RAG", version="3.19.0")

try:
    static_dir = os.path.dirname(os.path.abspath(__file__))
    app.mount("/static", StaticFiles(directory=static_dir, check_dir=True), name="static")
except Exception as e:
    log.warning(f"⚠️ Could not mount static files: {e}")

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

# ================== 7. Models ==================

class Message(BaseModel):
    role: str
    content: str

class AskBody(BaseModel):
    messages: List[Message]
    lang: str = "el"
    conversation_id: Optional[str] = None 

class TTSBody(BaseModel):
    text: str

class SurveyResponse(BaseModel):
    usedBot: str
    userStatus: Optional[str] = "N/A"
    scenarios: str
    gender: Optional[str] = "N/A"
    age: Optional[str] = "N/A"
    q1: int; q2: int; q3: int; q4: int; q5: int
    q6: int; q7: int; q8: int; q9: int; q10: int
    q11: int; q12: int; q13: int; q14: int; q15: int
    q16: int
    comments: Optional[str] = ""

# ================== 8. Helper: Auto-Log ==================

def log_chat_interaction(conv_id, user_q, bot_ans, ip):
    """Silently logs the chat turn to DB."""
    if not conv_id: return
    try:
        conn = get_db_conn()
        cur = conn.cursor()
        cur.execute(
            "INSERT INTO chatbot_feedback (conversation_id, user_question, bot_response, is_positive, ip_address) VALUES (%s, %s, %s, %s, %s)",
            (conv_id, user_q, bot_ans, None, ip)
        )
        conn.commit()
        cur.close()
        return_db_conn(conn)
    except Exception as e:
        log.error(f"Auto-log failed: {e}")

# ================== 9. Endpoints ==================

@app.get("/")
async def root(background_tasks: BackgroundTasks):
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
    target_lang = body.lang or "el"
    question = (body.messages[-1].content if body.messages else "").strip()
    if not question: return {"answer": "..."}
    
    client_ip = request.client.host

    # 1. LANG DETECT
    english_keywords = {'hello', 'hi', 'where', 'municipal', 'mayor', 'thank', 'when', 'what', 'how', 'who'}
    question_words = set(re.sub(r'[^\w\s]', '', question.lower()).split())
    if any(word in question_words for word in english_keywords): target_lang = 'en'
    else:
        try:
            if len(question) > 3:
                if detect(question) == 'en': target_lang = 'en'
        except: pass

    # 2. DIRECT ANSWER
    direct_resp = get_direct_answer(question)
    if direct_resp:
        async def direct_stream():
            ans = direct_resp["answer"]
            yield ans
            log_chat_interaction(body.conversation_id, question, ans, client_ip)
            
        return StreamingResponse(direct_stream(), media_type="text/plain")

    # 3. RAG SEARCH
    csv_context = ""
    def clean_text(t): return t.lower().translate(str.maketrans('', '', string.punctuation)).strip() if t else ""
    clean_user_q = clean_text(question)
    for row in knowledge_base:
        if len(row) >= 2:
            q_raw, a_val = list(row.values())[0], list(row.values())[1]
            if clean_text(str(q_raw)) in clean_user_q:
                csv_context += f"\nCSV Info: {a_val}\n"

    async def event_generator():
        conn = get_db_conn()
        full_response_accumulator = "" 
        try:
            cursor = conn.cursor()
            db_docs = retrieve_context(cursor, question, top_k=4)
            db_text = "\n".join([f"Info: {d['question']} - {d['answer']}" for d in db_docs])
            cursor.close()
            
            all_context = STATIC_KNOWLEDGE + "\n" + csv_context + "\n" + db_text
            
            # 4. SYSTEM PROMPT
            sys_msg = (
                f"You are Ephyra, the AI assistant for the Municipality of Corinth. "
                f"STRICT INSTRUCTIONS:\n"
                f"1. You MUST answer in the same language as the user's last message ({target_lang}).\n"
                f"2. You answer questions **ONLY** based on the provided CONTEXT below. **Do NOT use your internal training data, general knowledge, or internet info.**\n"
                f"3. **RESTRICTION:** If the user asks about general topics (e.g., gardening, cooking, world history, weather outside Corinth) that are NOT in the context, you MUST politely refuse.\n"
                f"4. **LISTING RULES:**\n"
                f"   - If the user asks for a SPECIFIC number (e.g. '3 places'), select exactly that many.\n"
                f"   - If the user asks GENERALLY (e.g. 'What can I see?', 'Suggest sights'), you MUST list **ALL** available options found in the Context. Do not summarize or select only a few.\n"
                f"5. **MUNICIPAL LINKS (MANDATORY):**\n"
                f"   - IF topic is **Registry / Birth Acts / Death Acts / Marriage Acts (Ληξιαρχείο)** -> Append: '\n🔗 Δήμος: https://korinthos.gr/odhgos-polith/vasikes-uphresies/lhksiarxeio/'\n"
                f"   - IF topic is **Certificates / Family Status / Birth Cert (Δημοτολόγιο)** -> Append: '\n🔗 Δήμος: https://korinthos.gr/odhgos-polith/vasikes-uphresies/dhmotologio/'\n"
                f"   - IF topic is **Civil Marriage (Πολιτικός Γάμος)** -> Append: '\n🔗 Δήμος: https://korinthos.gr/odhgos-polith/vasikes-uphresies/politiki-gamoi/'\n"
                f"   - IF topic is **Transfer (Μεταδημότευση)** -> Append: '\n🔗 Δήμος: https://korinthos.gr/odhgos-polith/vasikes-uphresies/dhmotologio/metadhmoteysh/'\n"
                f"6. **MITOS LOGIC:**\n"
                f"   - IF topic is a PROCEDURE, append: '\nΓια περισσότερες πληροφορίες μπορείτε να επισκεφθείτε και το mitos: https://mitos.gov.gr'.\n"
                f"   - **NEGATIVE:** IF asking for PHONES, HISTORY, SIGHTS, MAYOR, DEYA -> DO NOT append mitos.\n\n"
                f"CONTEXT:\n{all_context}"
            )
            
            response = openai_client.chat.completions.create(
                model="gpt-4o-mini",
                messages=[{"role": "system", "content": sys_msg}, {"role": "user", "content": question}],
                temperature=0.3, stream=True
            )
            for chunk in response:
                if chunk.choices[0].delta.content:
                    txt = chunk.choices[0].delta.content
                    full_response_accumulator += txt
                    yield txt
            
            log_chat_interaction(body.conversation_id, question, full_response_accumulator, client_ip)

        except Exception as e:
            log.error(f"Stream Error: {e}")
            yield "Sorry, connection error."
        finally:
            return_db_conn(conn)

    return StreamingResponse(event_generator(), media_type="text/plain")

@app.post("/feedback")
async def record_feedback(request: Request):
    try:
        data = await request.json()
        conn = get_db_conn(); cur = conn.cursor()
        # This endpoint is when user CLICKS thumbs up/down
        cur.execute("INSERT INTO chatbot_feedback (conversation_id, bot_response, user_question, is_positive, user_agent, ip_address) VALUES (%s, %s, %s, %s, %s, %s)",
            (data.get("conversation_id"), data.get("bot_response"), data.get("user_question"), data.get("is_positive"), request.headers.get("User-Agent"), request.client.host))
        conn.commit(); cur.close(); return_db_conn(conn)
        return {"status": "success"}
    except Exception as e: return {"error": str(e)}

@app.get("/feedback/stats")
async def get_feedback_stats(days: int = 30):
    conn = get_db_conn(); cur = conn.cursor()
    try:
        since_date = datetime.now() - timedelta(days=days)
        
        # ✨ BUG FIX: 
        # is_positive IS TRUE -> Positives
        # is_positive IS FALSE -> Negatives (Excludes NULLs)
        cur.execute("""
            SELECT 
                COUNT(*), 
                SUM(CASE WHEN is_positive IS TRUE THEN 1 ELSE 0 END), 
                SUM(CASE WHEN is_positive IS FALSE THEN 1 ELSE 0 END), 
                COUNT(DISTINCT ip_address) 
            FROM chatbot_feedback WHERE timestamp >= %s
        """, (since_date,))
        
        r = cur.fetchone()
        total, pos, neg, unique = r[0] or 0, r[1] or 0, r[2] or 0, r[3] or 0
        
        # Satisfaction Rate logic: (Pos / (Pos + Neg)) * 100
        rated_total = pos + neg
        satisfaction_rate = round((pos/rated_total*100)) if rated_total > 0 else 0

        cur.execute("SELECT DATE(timestamp), is_positive, COUNT(*) FROM chatbot_feedback WHERE timestamp >= %s GROUP BY DATE(timestamp), is_positive ORDER BY 1", (since_date,))
        daily = [{"date": str(row[0]), "sentiment": "positive" if row[1] else "negative", "count": row[2]} for row in cur.fetchall()]
        
        cur.execute("SELECT id, user_question, bot_response, is_positive, timestamp FROM chatbot_feedback WHERE timestamp >= %s ORDER BY timestamp DESC LIMIT 20", (since_date,))
        recent = [{"id":r[0], "user_question":r[1], "bot_response":r[2], "is_positive":r[3], "timestamp":str(r[4])} for r in cur.fetchall()]

        cur.execute("SELECT user_question, COUNT(*) as c FROM chatbot_feedback WHERE timestamp >= %s GROUP BY user_question ORDER BY c DESC LIMIT 5", (since_date,))
        top_qs = [{"question": r[0], "count": r[1]} for r in cur.fetchall()]

        cur.execute("SELECT user_question FROM chatbot_feedback WHERE timestamp >= %s", (since_date,))
        el_c, en_c = 0, 0
        for row in cur.fetchall():
            txt = (row[0] or "").strip()
            if not txt: continue
            if any('\u0370' <= c <= '\u03ff' or '\u1f00' <= c <= '\u1fff' for c in txt): el_c += 1
            else: en_c += 1

        return {
            "total_feedback": total, 
            "positive": pos, 
            "negative": neg, 
            "satisfaction_rate": satisfaction_rate, 
            "unique_users": unique, 
            "daily_data": daily, 
            "recent_feedback": recent, 
            "top_questions": top_qs, 
            "language_distribution": {"el": el_c, "en": en_c}
        }
    except Exception as e: return {"error": str(e)}
    finally: cur.close(); return_db_conn(conn)

@app.post("/submit_survey")
async def submit_survey(data: SurveyResponse):
    try:
        conn = get_db_conn(); cur = conn.cursor()
        cur.execute("""
            INSERT INTO survey_final_v2 
            (used_bot, user_status, scenarios_tested, gender, age, 
             q1, q2, q3, q4, q5, q6, q7, q8, q9, q10, 
             q11, q12, q13, q14, q15, q16, comments)
            VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s)
        """, (data.usedBot, data.userStatus, data.scenarios, data.gender, data.age, data.q1, data.q2, data.q3, data.q4, data.q5, data.q6, data.q7, data.q8, data.q9, data.q10, data.q11, data.q12, data.q13, data.q14, data.q15, data.q16, data.comments))
        conn.commit(); cur.close(); return_db_conn(conn)
        return {"status": "success"}
    except Exception as e: return {"error": str(e)}

@app.get("/survey_results")
async def get_survey_results():
    try:
        conn = get_db_conn(); cur = conn.cursor()
        cur.execute("SELECT * FROM survey_final_v2 ORDER BY timestamp DESC")
        cols = [desc[0] for desc in cur.description]
        results = [dict(zip(cols, row)) for row in cur.fetchall()]
        for r in results:
            if 'timestamp' in r and r['timestamp']: r['timestamp'] = str(r['timestamp'])
        cur.close(); return_db_conn(conn)
        return results
    except: return []

@app.get("/tts_play")
@app.post("/tts_play")
async def tts_play(request: Request, text: str = "", body: TTSBody = None):
    final_text = text
    if body and body.text: final_text = body.text
    if not final_text: return HTTPException(400)
    try:
        clean = final_text.replace("📍","").replace("📞","").strip()
        clean = re.sub(r'(\d)', r'\1 ', clean)
        audio = eleven_client.text_to_speech.convert(
            voice_id=ELEVENLABS_VOICE_ID, text=clean, model_id="eleven_multilingual_v2"
        )
        return StreamingResponse(io.BytesIO(b"".join(chunk for chunk in audio if chunk)), media_type="audio/mpeg")
    except: return HTTPException(500)

@app.post("/feedback/clear")
async def clear_all_data():
    conn = get_db_conn()
    try:
        cur = conn.cursor()
        cur.execute("TRUNCATE TABLE chatbot_feedback;")
        cur.execute("TRUNCATE TABLE survey_final_v2;")
        conn.commit(); cur.close()
        return {"status": "success"}
    except Exception as e: return {"error": str(e)}
    finally: return_db_conn(conn)

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)