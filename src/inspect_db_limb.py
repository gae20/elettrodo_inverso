import sqlite3, json

conn = sqlite3.connect(r'C:\Users\carme\THESIS\datasets\dataset\records.db')
cursor = conn.cursor()

# Keyword specificamente per inversioni periferiche/limb (escludendo precordiali V1-V6)
LIMB_INVERSION_KEYWORDS = [
    'inversione periferiche',
    'inversione periferica', 
    'inversion periferic',
    'elettrodi periferici',
    'elettrodi periferic',
    'scambio periferici',
    'scambio periferic',
    'inversione elettrodi periferic',
    'inversione degli elettrodi periferic',
    'inversione arti',
    'elettrodi arti',
    'scambio arti',
    'limb lead',
    'inversione limb',
    'inversione elettrodi',  # generico - può includere entrambi
]

# Keyword esplicitamente precordiali (da escludere)
PRECORDIAL_KEYWORDS = [
    'v1', 'v2', 'v3', 'v4', 'v5', 'v6',
    'precordiali', 'precordiale', 'precordial',
    'toracic',
]

cursor.execute("""
    SELECT id, status, text, report 
    FROM records 
    WHERE status='rejected'
""")
rows = cursor.fetchall()

limb_inversion_ids = []
precordial_only_ids = []
ambiguous_ids = []

for r in rows:
    id_ = r[0]
    raw_text = r[3]
    if isinstance(raw_text, bytes):
        raw_text = raw_text.decode('utf-8', errors='replace')
    text_str = (raw_text or '').lower()
    
    # Check if text mentions limb inversion keywords
    has_limb_kw = any(kw in text_str for kw in LIMB_INVERSION_KEYWORDS)
    # Also check generic inversion + scambio
    has_generic = ('inversion' in text_str or 'scambio' in text_str or 'elettrod' in text_str)
    has_prec = any(kw in text_str for kw in PRECORDIAL_KEYWORDS)
    
    if has_limb_kw and not has_prec:
        limb_inversion_ids.append((id_, text_str))
    elif has_limb_kw and has_prec:
        ambiguous_ids.append((id_, text_str))
    elif has_generic and not has_prec:
        ambiguous_ids.append((id_, text_str))

print(f"Limb inversion (clear, no precordial mentions): {len(limb_inversion_ids)}")
print(f"Ambiguous (mix or generic): {len(ambiguous_ids)}")
print("\n--- LIMB INVERSION EXAMPLES ---")
for id_, text in limb_inversion_ids[:20]:
    print(f"  ID={id_}: {text[:200]}")

print("\n--- AMBIGUOUS EXAMPLES ---")
for id_, text in ambiguous_ids[:20]:
    print(f"  ID={id_}: {text[:200]}")

conn.close()
