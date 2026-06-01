import zipfile, sqlite3

# Vedi come sono nominati i file nello ZIP
zip_path = r'C:\Users\carme\THESIS\datasets\dataset\DATASET_complete\dataset_batch_1.zip'
with zipfile.ZipFile(zip_path, 'r') as z:
    names = z.namelist()[:10]
    print('File names in ZIP (first 10):')
    for n in names:
        print(' ', repr(n))

# Vedi come sono fatti gli ID nel DB
conn = sqlite3.connect(r'C:\Users\carme\THESIS\datasets\dataset\records.db')
cursor = conn.cursor()
cursor.execute("SELECT id FROM records WHERE status='rejected' LIMIT 5")
rows = cursor.fetchall()
print('\nID nel DB (prime 5):')
for r in rows:
    print(' ', repr(r[0]))
conn.close()
