import pandas as pd
import psycopg2
import os
from dotenv import load_dotenv

load_dotenv()

def import_csv():
    try:
        conn = psycopg2.connect(
            dbname=os.getenv("DB_NAME"),
            user=os.getenv("DB_USER"),
            password=os.getenv("DB_PASS"),
            host=os.getenv("DB_HOST"),
            port=os.getenv("DB_PORT", "5432")
        )
        cur = conn.cursor()
        csv_file = "QA_chatbot(gr).csv"
        
        # Διαβάζουμε το CSV με το ερωτηματικό (;) που έχει το αρχείο σου
        df = pd.read_csv(csv_file, sep=';', encoding='utf-8')
        print(f"Ανάγνωση αρχείου: {csv_file}...")
        print(f"Βρέθηκαν {len(df)} γραμμές.")

        # --- ΑΝΤΙΚΑΤΑΣΤΑΣΗ ΕΔΩ ---
        for index, row in df.iterrows():
            # Χρησιμοποιούμε iloc για να παίρνουμε τα δεδομένα με τη σωστή σειρά στηλών:
            # 1η στήλη (index 0) = intent (π.χ. city_history)
            # 2η στήλη (index 1) = question (Η ερώτηση)
            # 3η στήλη (index 2) = answer (Η απάντηση)
            sql_intent = str(row.iloc[0]).strip() 
            sql_question = str(row.iloc[1]).strip() 
            sql_answer = str(row.iloc[2]).strip()

            cur.execute(
                "INSERT INTO kb_items_raw (question, answer, intent) VALUES (%s, %s, %s)",
                (sql_question, sql_answer, sql_intent)
            )
        # --- ΤΕΛΟΣ ΑΝΤΙΚΑΤΑΣΤΑΣΗΣ ---
        
        conn.commit()
        cur.close()
        conn.close()
        print("✅ Τα δεδομένα μπήκαν ΣΩΣΤΑ στις στήλες τους!")
    except Exception as e:
        print(f"❌ Σφάλμα: {e}")

if __name__ == "__main__":
    import_csv()