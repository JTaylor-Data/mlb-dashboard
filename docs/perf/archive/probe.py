import sqlite3, os
conn = sqlite3.connect("mlb_data.db")
cur = conn.cursor()
for t in ["pitches","players","teams","schedule_games","rosters"]:
    try:
        cur.execute(f"SELECT COUNT(*) FROM {t}")
        print(f"{t}: {cur.fetchone()[0]:,}")
    except Exception as e:
        print(f"{t}: error - {e}")
print(f"DB size: {os.path.getsize('mlb_data.db')/1e9:.2f} GB")
cur.execute("SELECT name,tbl_name FROM sqlite_master WHERE type='index' ORDER BY tbl_name")
rows = cur.fetchall()
print(f"\nExisting indexes ({len(rows)}):")
for r in rows:
    print(f"  {r[1]}.{r[0]}")
conn.close()
