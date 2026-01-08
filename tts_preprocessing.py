"""
TTS Preprocessing για Ελληνικά
===============================

Κανονικοποιεί κείμενο για καλύτερη προφορά από TTS συστήματα (ElevenLabs)

Χειρίζεται:
- Ελληνικούς αριθμούς (15 → δεκαπέντε)
- Τηλέφωνα (27410-76353 → δύο επτά τέσσερα ένα μηδέν, επτά έξι τρία πέντε τρία)
- Email (info@corinth.gr → info σε corinth τελεία gr)
- URLs
- Νομίσματα (15€ → δεκαπέντε ευρώ)
- Ώρες (09:00 → εννιά το πρωί)
"""

import re
from typing import Dict

# ============================================================================
# ΑΡΙΘΜΟΙ
# ============================================================================

# Αριθμοί 0-19
UNITS = {
    0: "μηδέν", 1: "ένα", 2: "δύο", 3: "τρία", 4: "τέσσερα", 
    5: "πέντε", 6: "έξι", 7: "επτά", 8: "οκτώ", 9: "εννιά",
    10: "δέκα", 11: "έντεκα", 12: "δώδεκα", 13: "δεκατρία", 
    14: "δεκατέσσερα", 15: "δεκαπέντε", 16: "δεκαέξι", 
    17: "δεκαεπτά", 18: "δεκαοκτώ", 19: "δεκαεννιά"
}

# Δεκάδες
TENS = {
    20: "είκοσι", 30: "τριάντα", 40: "σαράντα", 50: "πενήντα",
    60: "εξήντα", 70: "εβδομήντα", 80: "ογδόντα", 90: "ενενήντα"
}

# Εκατοντάδες
HUNDREDS = {
    100: "εκατό", 200: "διακόσια", 300: "τριακόσια", 400: "τετρακόσια",
    500: "πεντακόσια", 600: "εξακόσια", 700: "επτακόσια", 
    800: "οκτακόσια", 900: "εννιακόσια"
}


def number_to_greek(n: int) -> str:
    """
    Μετατρέπει αριθμό σε ελληνική λέξη
    
    Παραδείγματα:
        15 → "δεκαπέντε"
        123 → "εκατόν είκοσι τρία"
        1000 → "χίλια"
    """
    if n == 0:
        return "μηδέν"
    
    if n < 0:
        return "μείον " + number_to_greek(-n)
    
    # 1-19
    if n < 20:
        return UNITS[n]
    
    # 20-99
    if n < 100:
        tens = (n // 10) * 10
        units = n % 10
        if units == 0:
            return TENS[tens]
        return f"{TENS[tens]} {UNITS[units]}"
    
    # 100-999
    if n < 1000:
        hundreds = (n // 100) * 100
        remainder = n % 100
        
        if remainder == 0:
            return HUNDREDS[hundreds]
        
        if hundreds == 100:
            return f"εκατόν {number_to_greek(remainder)}"
        else:
            return f"{HUNDREDS[hundreds]} {number_to_greek(remainder)}"
    
    # 1000-9999
    if n < 10000:
        thousands = n // 1000
        remainder = n % 1000
        
        if thousands == 1:
            thousands_word = "χίλια"
        elif thousands == 2:
            thousands_word = "δύο χιλιάδες"
        else:
            thousands_word = f"{number_to_greek(thousands)} χιλιάδες"
        
        if remainder == 0:
            return thousands_word
        return f"{thousands_word} {number_to_greek(remainder)}"
    
    # 10000+
    return str(n)  # Fallback για πολύ μεγάλους αριθμούς


# ============================================================================
# ΤΗΛΕΦΩΝΑ
# ============================================================================

def normalize_phone(phone: str) -> str:
    """
    Μετατρέπει τηλέφωνα σε προφορά
    
    Παραδείγματα:
        "27410-76353" → "δύο επτά τέσσερα ένα μηδέν, επτά έξι τρία πέντε τρία"
        "2741076353" → "δύο επτά τέσσερα ένα μηδέν επτά έξι τρία πέντε τρία"
        "6912345678" → "έξι εννιά ένα δύο τρία τέσσερα πέντε έξι επτά οκτώ"
    """
    # Αφαίρεση ειδικών χαρακτήρων
    clean = re.sub(r'[^\d]', '', phone)
    
    # Διάβασε ψηφίο-ψηφίο
    digits = []
    for digit in clean:
        digits.append(UNITS[int(digit)])
    
    # Αν έχει παύλα ή κενό στο original, κάνε παύση
    if '-' in phone or ' ' in phone:
        # Βρες τη θέση της παύλας
        dash_pos = phone.find('-') if '-' in phone else phone.find(' ')
        # Μέτρα πόσα ψηφία υπάρχουν πριν
        digits_before_dash = len(re.sub(r'[^\d]', '', phone[:dash_pos]))
        
        # Βάλε κόμμα στη σωστή θέση
        result = ' '.join(digits[:digits_before_dash])
        result += ', '
        result += ' '.join(digits[digits_before_dash:])
        return result
    
    return ' '.join(digits)


# ============================================================================
# EMAIL
# ============================================================================

def normalize_email(email: str) -> str:
    """
    Μετατρέπει email σε προφορά
    
    Παραδείγματα:
        "info@corinth.gr" → "info σε corinth τελεία gr"
        "mayor@dimoskorinthos.gov.gr" → "mayor σε dimoskorinthos τελεία gov τελεία gr"
    """
    # Αντικατάσταση @ με "σε"
    result = email.replace('@', ' σε ')
    
    # Αντικατάσταση . με "τελεία"
    result = result.replace('.', ' τελεία ')
    
    return result


# ============================================================================
# URLs
# ============================================================================

def normalize_url(url: str) -> str:
    """
    Μετατρέπει URL σε προφορά
    
    Παραδείγματα:
        "www.corinth.gr" → "www τελεία corinth τελεία gr"
        "https://dimoskorinthos.gov.gr" → "https dimoskorinthos τελεία gov τελεία gr"
    """
    # Αφαίρεση protocol
    result = re.sub(r'https?://', '', url)
    
    # Αντικατάσταση . με "τελεία"
    result = result.replace('.', ' τελεία ')
    
    # Αντικατάσταση / με "κάθετο"
    result = result.replace('/', ' κάθετο ')
    
    return result


# ============================================================================
# ΝΟΜΙΣΜΑΤΑ
# ============================================================================

def normalize_currency(text: str) -> str:
    """
    Μετατρέπει νομίσματα σε προφορά
    
    Παραδείγματα:
        "15€" → "δεκαπέντε ευρώ"
        "€15" → "δεκαπέντε ευρώ"
        "100 ευρώ" → "εκατό ευρώ"
    """
    # Pattern 1: Αριθμός + € (π.χ. 15€)
    text = re.sub(r'(\d+)\s*€', lambda m: f"{number_to_greek(int(m.group(1)))} ευρώ", text)
    
    # Pattern 2: € + Αριθμός (π.χ. €15)
    text = re.sub(r'€\s*(\d+)', lambda m: f"{number_to_greek(int(m.group(1)))} ευρώ", text)
    
    return text


# ============================================================================
# ΩΡΕΣ
# ============================================================================

def normalize_time(text: str) -> str:
    """
    Μετατρέπει ώρες σε προφορά
    
    Παραδείγματα:
        "09:00" → "εννιά το πρωί"
        "14:30" → "δύο και μισή το μεσημέρι"
        "18:00" → "έξι το απόγευμα"
    """
    # Pattern: HH:MM
    pattern = r'\b(\d{1,2}):(\d{2})\b'
    
    def replace_time(match):
        hour = int(match.group(1))
        minute = int(match.group(2))
        
        # Μετατροπή σε 12ωρο
        period = ""
        if 5 <= hour < 12:
            period = " το πρωί"
        elif 12 <= hour < 18:
            period = " το μεσημέρι"
        elif 18 <= hour < 21:
            period = " το απόγευμα"
        else:
            period = " το βράδυ"
        
        # Ώρα
        if hour > 12:
            hour -= 12
        elif hour == 0:
            hour = 12
        
        hour_text = number_to_greek(hour)
        
        # Λεπτά
        if minute == 0:
            return f"{hour_text}{period}"
        elif minute == 30:
            return f"{hour_text} και μισή{period}"
        elif minute == 15:
            return f"{hour_text} και τέταρτο{period}"
        else:
            minute_text = number_to_greek(minute)
            return f"{hour_text} και {minute_text}{period}"
    
    return re.sub(pattern, replace_time, text)


# ============================================================================
# ΚΥΡΙΑ ΣΥΝΑΡΤΗΣΗ
# ============================================================================

def preprocess_for_tts(text: str) -> str:
    """
    Κανονικοποιεί κείμενο για TTS
    
    Εφαρμόζει όλες τις μετατροπές με τη σωστή σειρά
    """
    result = text
    
    # 1. Emails (πριν τα URLs γιατί μπορεί να έχουν @ και .)
    result = re.sub(
        r'\b[a-zA-Z0-9._%+-]+@[a-zA-Z0-9.-]+\.[a-zA-Z]{2,}\b',
        lambda m: normalize_email(m.group(0)),
        result
    )
    
    # 2. URLs
    result = re.sub(
        r'\b(?:https?://)?(?:www\.)?[a-zA-Z0-9-]+\.[a-zA-Z]{2,}(?:/[^\s]*)?\b',
        lambda m: normalize_url(m.group(0)),
        result
    )
    
    # 3. Τηλέφωνα (10ψήφια με ή χωρίς παύλες/κενά)
    result = re.sub(
        r'\b\d{5}[-\s]?\d{5}\b',
        lambda m: normalize_phone(m.group(0)),
        result
    )
    result = re.sub(
        r'\b\d{10}\b',
        lambda m: normalize_phone(m.group(0)),
        result
    )
    
    # 4. Ώρες (HH:MM)
    result = normalize_time(result)
    
    # 5. Νομίσματα
    result = normalize_currency(result)
    
    # 6. Standalone αριθμοί (μέχρι 4 ψηφία)
    # Προσοχή: Μόνο standalone, όχι μέσα σε λέξεις
    result = re.sub(
        r'\b(\d{1,4})\b',
        lambda m: number_to_greek(int(m.group(1))),
        result
    )
    
    # 7. Καθαρισμός διπλών κενών
    result = re.sub(r'\s+', ' ', result)
    
    return result.strip()


# ============================================================================
# TESTS
# ============================================================================

if __name__ == "__main__":
    print("=" * 70)
    print("TTS PREPROCESSING TESTS")
    print("=" * 70)
    
    test_cases = [
        # Αριθμοί
        ("Κοστίζει 15 ευρώ", "Κοστίζει δεκαπέντε ευρώ"),
        ("Υπάρχουν 123 πολίτες", "Υπάρχουν εκατόν είκοσι τρία πολίτες"),
        
        # Τηλέφωνα
        ("Καλέστε στο 27410-76353", "Καλέστε στο δύο επτά τέσσερα ένα μηδέν, επτά έξι τρία πέντε τρία"),
        ("Κινητό: 6912345678", "Κινητό: έξι εννιά ένα δύο τρία τέσσερα πέντε έξι επτά οκτώ"),
        
        # Email
        ("Email: info@corinth.gr", "Email: info σε corinth τελεία gr"),
        
        # URLs
        ("Επισκεφθείτε www.corinth.gr", "Επισκεφθείτε www τελεία corinth τελεία gr"),
        
        # Ώρες
        ("Ανοίγουμε στις 09:00", "Ανοίγουμε στις εννιά το πρωί"),
        ("Κλείνουμε στις 14:30", "Κλείνουμε στις δύο και μισή το μεσημέρι"),
        
        # Νομίσματα
        ("Κοστίζει 50€", "Κοστίζει πενήντα ευρώ"),
        ("€100", "εκατό ευρώ"),
        
        # Mixed
        ("Καλέστε στο 27410-76353 ή στείλτε email στο info@corinth.gr. Ώρες: 09:00-14:00",
         "Καλέστε στο δύο επτά τέσσερα ένα μηδέν, επτά έξι τρία πέντε τρία ή στείλτε email στο info σε corinth τελεία gr. Ώρες: εννιά το πρωί-δύο το μεσημέρι"),
    ]
    
    print("\n")
    for i, (input_text, expected) in enumerate(test_cases, 1):
        result = preprocess_for_tts(input_text)
        status = "✅" if result == expected else "❌"
        
        print(f"Test {i}: {status}")
        print(f"  Input:    {input_text}")
        print(f"  Output:   {result}")
        if result != expected:
            print(f"  Expected: {expected}")
        print()
