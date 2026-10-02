import sqlite3, os
os.chdir(r"C:\Users\joshs\Desktop\mlb_dashboard")
for db in ["mlb_data.db", "data/mlb.db"]:
    try:
        conn = sqlite3.connect(db)
        c = conn.cursor()
        c.execute("SELECT name FROM sqlite_master WHERE type='table'")
        tables = [r[0] for r in c.fetchall()]
        print(f"{db}: tables = {tables}")
        conn.close()
    except Exception as e:
        print(f"{db}: ERROR {e}")
