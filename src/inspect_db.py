import sqlite3

conn = sqlite3.connect(r'C:\Users\carme\THESIS\datasets\dataset\records.db')
cursor = conn.cursor()

cursor.execute('PRAGMA table_info(records)')
cols = cursor.fetchall()
print("COLUMNS:")
for c in cols:
    print(c)

print('---')
cursor.execute("SELECT COUNT(*) FROM records WHERE status='rejected'")
print('Total rejected:', cursor.fetchone()[0])

cursor.execute("SELECT COUNT(*) FROM records")
print('Total records:', cursor.fetchone()[0])

cursor.execute("SELECT DISTINCT status FROM records")
print('Statuses:', cursor.fetchall())

# Cerca ECG con testo che contenga inversione nei rejected
cursor.execute("""
    SELECT id, status, text, report 
    FROM records 
    WHERE status='rejected' 
    AND (
        LOWER(text) LIKE '%inversion%'
        OR LOWER(text) LIKE '%inversione%'
        OR LOWER(text) LIKE '%scambio%'
        OR LOWER(text) LIKE '%periferic%'
        OR LOWER(text) LIKE '%elettrod%'
        OR LOWER(text) LIKE '%limb%'
    )
    LIMIT 10
""")
rows = cursor.fetchall()
print(f"\nRejected + inversione keywords: found {len(rows)} (limited to 10)")
for r in rows:
    print(f"  ID={r[0]}, status={r[1]}, text={r[2][:200] if r[2] else None}")

# Count totale
cursor.execute("""
    SELECT COUNT(*)
    FROM records 
    WHERE status='rejected' 
    AND (
        LOWER(text) LIKE '%inversion%'
        OR LOWER(text) LIKE '%inversione%'
        OR LOWER(text) LIKE '%scambio%'
        OR LOWER(text) LIKE '%periferic%'
        OR LOWER(text) LIKE '%elettrod%'
        OR LOWER(text) LIKE '%limb%'
    )
""")
print("Total count:", cursor.fetchone()[0])

conn.close()
