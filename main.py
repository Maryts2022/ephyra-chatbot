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
from fastapi import FastAPI, HTTPException, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import StreamingResponse, FileResponse
from fastapi.staticfiles import StaticFiles

app = FasstAPI()
#Εδώ ακριβώς κόλλησε το mount:
app.mount("/static", StaticFiles(directory="static"), name="static")

from pydantic import BaseModel
from dotenv import load_dotenv
from slowapi import Limiter, _rate_limit_exceeded_handler
from slowapi.util import get_remote_address
from slowapi.errors import RateLimitExceeded

from openai import OpenAI
from elevenlabs.client import ElevenLabs
import re

from sentence_transformers import SentenceTransformer
from langdetect import detect, LangDetectException
from fastapi.middleware.cors import CORSMiddleware
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

def get_db_conn():
    try:
        return conn_pool.getconn()
    except Exception as e:
        log.exception("❌ Failed to get DB connection")
        raise

def return_db_conn(conn):
    if conn:
        conn_pool.putconn(conn)

# Alias για συμβατότητα
get_db_connection = get_db_conn

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

def retrieve_context(cursor, question: str, top_k: int = 10) -> List[Dict]:
    """
    RAG Step 1: RETRIEVE
    Combines semantic + keyword search, ranks and deduplicates results.
    Returns MORE documents so Gemini can find the best match regardless of wording.
    """
   # 1. Εκτέλεση και των δύο αναζητήσεων (Semantic & Keyword)
    # Αυξάνουμε το top_k για να έχουμε μεγαλύτερη πιθανότητα να βρούμε την πληροφορία
    semantic_results = semantic_search(cursor, question, top_k=20)
    top_sim = semantic_results[0]['similarity'] if semantic_results else 0
    
    keyword_results = keyword_search(cursor, question, top_k=20)

    # 2. ΥΒΡΙΔΙΚΟΣ ΣΥΝΔΥΑΣΜΟΣ (Hybrid Search) - ΧΩΡΙΣ ΠΕΡΙΟΡΙΣΜΟΥΣ
    # Χρησιμοποιούμε dictionary για να ενώσουμε τα αποτελέσματα χωρίς διπλότυπα
    all_results = {}
    
    # Πρώτα προσθέτουμε τα αποτελέσματα από την αναζήτηση λέξεων-κλειδιών
    for doc in keyword_results:
        all_results[doc['id']] = doc
    
    # Μετά προσθέτουμε τα αποτελέσματα από τη σημασιολογική αναζήτηση
    for doc in semantic_results:
        if doc['id'] not in all_results:
            all_results[doc['id']] = doc
        else:
            # Αν ένα έγγραφο βρέθηκε και από τους δύο τρόπους, κρατάμε το υψηλότερο similarity
            current_sim = doc.get('similarity', 0)
            existing_sim = all_results[doc['id']].get('similarity', 0)
            all_results[doc['id']]['similarity'] = max(current_sim, existing_sim)
    
    # Καταγραφή πληροφοριών στο Log για έλεγχο στο τερματικό σου
    log.info(f"📂 Hybrid Search: Top Semantic Sim: {top_sim:.3f}")
    log.info(f"📚 Total unique documents collected for GPT: {len(all_results)}")

    # 3. Ταξινόμηση βάσει συνάφειας και επιστροφή
    # Χρησιμοποιούμε το top_k που έχει οριστεί (συνήθως 10-15 για το GPT)
    ranked = sorted(all_results.values(), key=lambda x: x.get('similarity', 0), reverse=True)[:top_k]
    
    # ΕΠΙΣΤΡΟΦΗ ΤΗΣ ΛΙΣΤΑΣ ΣΤΗ ΣΥΝΑΡΤΗΣΗ generate_answer_with_rag
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

async def generate_answer_with_rag(question: str, context_docs: List[Dict], 
                                   lang: str = "el", conversation_history: List[Message] = None) -> Tuple[str, Dict]:
    """
    Παραγωγή AI απάντησης με πλήρη αξιοποίηση του GPT-4o-mini και των 88 πληροφοριών.
    """
    log.info(f"🤖 Generating AI response in '{lang}' (Found {len(context_docs)} relevant docs)...")
    
    # 1. Βελτιωμένο Formatting του Context για να το "βλέπει" καθαρά το GPT
    if not context_docs:
        context_str = "Δεν βρέθηκαν σχετικά έγγραφα στη βάση δεδομένων."
    else:
        # Ενώνουμε τις πληροφορίες σε μια καθαρή λίστα
         # Μετατροπή των docs σε κείμενο που διαβάζει η AI
          context_str = "\n".join([f"Ερώτηση: {d.get('question', '')} - Απάντηση: {d.get('answer', '')}" for d in context_docs])

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
    
    "2. ΑΚΡΙΒΕΙΑ ΤΗΛΕΦΩΝΩΝ: Μην επινοείς ποτέ τηλεφωνικά νούμερα. Αν ο χρήστης ρωτάει για βλάβες ή υπηρεσίες, "
    "δώσε το ακριβές νούμερο που αναγράφεται στο αντίστοιχο έγγραφο (π.χ. για ηλεκτροφωτισμό το 2741120134). "
    "Αν δεν υπάρχει ειδικό νούμερο, χρησιμοποίησε το γενικό κέντρο 2741361000.\n"
    
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
        # 3. Κλήση OpenAI με GPT-4o-mini
        response = openai_client.chat.completions.create(
            model="gpt-4o-mini",
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "system", "content": f"ΔΙΑΘΕΣΙΜΕΣ ΠΛΗΡΟΦΟΡΙΕΣ (CONTEXT):\n{context_str}"},
                {"role": "user", "content": question}
            ],
            temperature=0.7 # Για φυσικότητα στον λόγο
        )
        answer = response.choices[0].message.content.strip()
        
        # 4. Metadata για το UI
        metadata = {
            "documents_used": len(context_docs),
            "avg_similarity": sum(d.get('similarity', 0) for d in context_docs) / len(context_docs) if context_docs else 0,
            "sources": list(set(d.get('source', 'Βάση Δεδομένων') for d in context_docs))
        }
        return answer, metadata

    except Exception as e:
        log.error(f"❌ OpenAI Error: {e}")
        error_msg = "Λυπάμαι, παρουσιάστηκε σφάλμα στη σύνδεση με το AI." if lang == "el" else "Sorry, an AI error occurred."
        return error_msg, {"error": str(e)}

# ================== Endpoints ==================

@app.get("/")
async def root():
    """Serve HTML UI."""
    current_dir = os.path.dirname(os.path.abspath(__file__))
    html_path = os.path.join(current_dir, "ui_chatbot.html")
    
    if os.path.exists(html_path):
        return FileResponse(html_path, media_type="text/html")
    
    return {"message": "Ephyra Chatbot API v3.0.0 - RAG Edition"}

@app.get("/health")
async def health():
    """Health check endpoint."""
    return {
        "status": "healthy",
        "version": "3.0.0-RAG",
        "architecture": "Retrieval Augmented Generation with OpenAI GPT-4o-mini"
    }

@app.post("/ask")
@limiter.limit("30/minute")
async def ask(request: Request, body: AskBody):
    """
    Main endpoint: Implements full RAG pipeline
    1. RETRIEVE: Get context from semantic + keyword search
    2. AUGMENT: Format context
    3. GENERATE: Use Gemini to create natural response
    """
    conn = None
    current_lang = body.lang or detect_user_lang(body.messages[-1].content if body.messages else "")
    question = (body.messages[-1].content if body.messages else "").strip()

    try:
        if not question:
            return {
                "answer": "❌ No question received" if current_lang == 'en' else "❌ Δεν έλαβα ερώτηση",
                "quality": "error"
            }
        
        # Check for direct answers
        direct_answer = get_direct_answer(question)
        if direct_answer:
            log.info(f"✓ Direct answer matched: {question}")
            return direct_answer
        
        # Check for capabilities question
        if is_capabilities_question(question):
            log.info(f"ℹ️ Capabilities question detected: {question}")
            capabilities = get_capabilities_response(current_lang)
            return {
                "answer": capabilities,
                "quality": "capabilities",
                "context_found": True,
                "confidence": 1.0
            }
        
        # Check for greeting
        if is_greeting(question):
            if len(body.messages) <= 1:
                log.info(f"👋 Greeting detected on first message: {question}")
                if current_lang == 'en':
                    greeting = "Hello! I'm Ephyra, the professional AI assistant for the Municipality of Corinth. How can I help you today? 😊"
                else:
                    greeting = "Γεια σας! Είμαι η Εφύρα, η επαγγελματική AI βοηθός του Δήμου Κορινθίων. Πώς μπορώ να σας βοηθήσω σήμερα; 😊"
                
                return {
                    "answer": greeting,
                    "quality": "greeting",
                    "context_found": False,
                    "confidence": 1.0
                }
            else:
                log.info(f"👋 Greeting detected (not first message, skipping greeting response)")
                pass
        
        # Check for out of scope
        if is_out_of_scope(question):
            log.info(f"⛔ Out of scope: {question[:50]}")
            if current_lang == 'en':
                msg = ("I'm sorry, I only assist with questions about the Municipality of Corinth. "
                      "For other topics, please ask something related to municipal services.")
            else:
                msg = ("Λυπάμαι, βοηθώ μόνο με ερωτήσεις σχετικά με το Δήμο Κορινθίων. "
                      "Για άλλα θέματα, παρακαλώ ρωτήστε κάτι σχετικό με τις δημοτικές υπηρεσίες.")
            
            return {
                "answer": msg,
                "quality": "out_of_scope",
                "context_found": False,
                "confidence": 0.0
            }
        
        # ==================== RAG PIPELINE ====================
        # Ανοίγουμε τη σύνδεση ΜΕΣΑ στο try
        conn = get_db_conn()
        cursor = conn.cursor()
        
        # Step 1: RETRIEVE context
        context_docs = retrieve_context(cursor, question, top_k=body.top_k)
        
        # Κλείνουμε τον cursor αμέσως μετά την ανάκτηση
        cursor.close()
        
        # Step 3: GENERATE answer
        answer, metadata = await generate_answer_with_rag(
            question, 
            context_docs, 
            current_lang,
            body.messages
        )
        
        # Calculate final confidence
        confidence = metadata.get('avg_similarity', 0.0) if context_docs else 0.0
        
        return {
            "answer": answer,
            "quality": "generated",
            "context_found": len(context_docs) > 0,
            "confidence": float(confidence),
            "documents_used": metadata.get('documents_used', 0),
            "sources": metadata.get('sources', [])
        }
    
    except Exception as e:
        error_msg_full = f"❌ ERROR in /ask: {str(e)}"
        log.exception(error_msg_full)
        print(f"\n🔴 CRITICAL ERROR: {error_msg_full}")
        print(f"   Exception type: {type(e).__name__}")
        print(f"   Question was: {question if 'question' in locals() else 'N/A'}")
        import traceback
        traceback.print_exc()
        
        if current_lang == 'en':
            error_msg = "An unexpected error occurred. Please try again."
        else:
            error_msg = "Παρουσιάστηκε απρόσμενο σφάλμα. Παρακαλώ δοκιμάστε ξανά."
        
        return {
            "answer": error_msg,
            "quality": "error",
            "context_found": False,
            "confidence": 0.0
        }
    
    finally:
        # Η επιστροφή της σύνδεσης γίνεται ΠΑΝΤΑ εδώ
        if conn:
            return_db_conn(conn)
            log.info("🔌 Connection returned to pool successfully.")

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
    try:
        conn = get_db_connection()
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
            SELECT COUNT(*) as count
            FROM chatbot_feedback
            WHERE timestamp >= %s
        """, (since_date,))
        
        language_distribution = {"el": 100, "en": 0}
        
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
        cursor.execute("""
            SELECT id, user_question, bot_response, is_positive, timestamp, ip_address
            FROM chatbot_feedback
            WHERE timestamp >= %s
            ORDER BY timestamp DESC
            LIMIT 50
        """, (since_date,))
        
        recent_feedback = []
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
        conn.close()
        
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
            "recent_feedback": recent_feedback if detailed else []
        }
        
    except Exception as e:
        log.error(f"Error getting feedback stats: {e}")
        return {"error": str(e)}
        
        top_issues = [{"category": row[0], "count": row[1]} for row in cursor.fetchall()]
        
        # Most frequent questions
        cursor.execute("""
            SELECT user_question, COUNT(*) as count
            FROM chatbot_feedback
            WHERE timestamp >= %s AND user_question IS NOT NULL AND user_question != ''
            GROUP BY user_question
            ORDER BY count DESC
            LIMIT 5
        """, (since_date,))
        
        top_questions = [{"question": row[0], "count": row[1]} for row in cursor.fetchall()]
        
        # Language distribution
        cursor.execute("""
            SELECT 
                CASE WHEN user_question ILIKE '%[αβγδεζηθικλμνξοπρστυφχψω]%' THEN 'el' ELSE 'en' END as lang,
                COUNT(*) as count
            FROM chatbot_feedback
            WHERE timestamp >= %s
            GROUP BY lang
        """, (since_date,))
        
        lang_dist = {row[0]: row[1] for row in cursor.fetchall()}
        total_lang = sum(lang_dist.values())
        language_distribution = {k: round(v/total_lang*100) if total_lang > 0 else 0 
                                for k, v in lang_dist.items()}
        
        # Sentiment trend (compare with previous period)
        prev_since = since_date - timedelta(days=days)
        cursor.execute("""
            SELECT 
                SUM(CASE WHEN is_positive THEN 1 ELSE 0 END) as positive,
                COUNT(*) as total
            FROM chatbot_feedback
            WHERE timestamp >= %s AND timestamp < %s
        """, (prev_since, since_date))
        
        prev_row = cursor.fetchone()
        prev_satisfaction = round((prev_row[0] / prev_row[1] * 100)) if prev_row[1] > 0 else 0
        sentiment_trend = satisfaction_rate - prev_satisfaction
        
        # User metrics
        cursor.execute("""
            SELECT 
                COUNT(DISTINCT ip_address) as unique_users,
                COUNT(DISTINCT CASE WHEN timestamp >= NOW() - interval '7 days' THEN ip_address END) as weekly_active
            FROM chatbot_feedback
            WHERE timestamp >= %s
        """, (since_date,))
        
        user_row = cursor.fetchone()
        unique_users = user_row[0] if user_row[0] else 0
        weekly_active = user_row[1] if user_row[1] else 0
        
        # Recent feedback
        cursor.execute("""
            SELECT id, user_question, bot_response, is_positive, timestamp, ip_address, 
                   CASE WHEN user_question ILIKE '%[αβγδεζηθικλμνξοπρστυφχψω]%' THEN 'el' ELSE 'en' END,
            FROM chatbot_feedback
            WHERE timestamp >= %s
            ORDER BY timestamp DESC
            LIMIT 50
        """, (since_date,))
        
        recent_feedback = [
            {
                "id": row[0],
                "user_question": row[1],
                "bot_response": row[2],
                "is_positive": row[3],
                "timestamp": row[4].isoformat() if row[4] else None,
                "ip_address": row[5],
                "language": row[6],
                "response_time": row[7]
            }
            for row in cursor.fetchall()
        ]
        
        cursor.close()
        conn.close()
        
        return {
            "total_feedback": total,
            "positive": positive,
            "negative": negative,
            "satisfaction_rate": satisfaction_rate,
            "daily_data": daily_data,
            "top_issues": top_issues,
            "top_questions": top_questions,
            "language_distribution": language_distribution,
            "sentiment_trend": sentiment_trend,
            "sentiment_trend_percent": f"+{sentiment_trend}%" if sentiment_trend > 0 else f"{sentiment_trend}%",
            "avg_response_time": avg_response_time,
            "min_response_time": min_response_time,
            "max_response_time": max_response_time,
            "unique_users": unique_users,
            "weekly_active_users": weekly_active,
            "recent_feedback": recent_feedback if detailed else []
        }
        
    except Exception as e:
        log.error(f"Error getting feedback stats: {e}")
        return {"error": str(e)}


@app.get("/feedback/export")
async def export_feedback(days: int = 30, format: str = "csv"):
    """Export feedback data."""
    try:
        conn = get_db_connection()
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
        conn = get_db_connection()
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

if __name__ == "__main__":
    import uvicorn
    log.info("🚀 Ephyra Chatbot v3.0.0 - Production RAG Edition starting...")
    uvicorn.run(app, host="0.0.0.0", port=8000)
