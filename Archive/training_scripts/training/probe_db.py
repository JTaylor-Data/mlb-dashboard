import os, sqlite3, json

for candidate in ["data/mlb.db", "mlb_data.db", "data/mlb_data.db"]:
    if os.path.exists(candidate) and os.path.getsize(candidate) > 1000:
        db = candidate
        break

conn = sqlite3.connect(db)
c = conn.cursor()

# Full pitches schema
c.execute("PRAGMA table_info(pitches)")
cols = [(r[1], r[2]) for r in c.fetchall()]
print("PITCHES COLUMNS:")
for name, typ in cols:
    print(f"  {name}: {typ}")

print()
# player_advanced_stats full schema
c.execute("PRAGMA table_info(player_advanced_stats)")
cols = [(r[1], r[2]) for r in c.fetchall()]
print("PLAYER_ADVANCED_STATS COLUMNS:")
for name, typ in cols:
    print(f"  {name}: {typ}")

print()
# pitcher_model_scores season range
c.execute("SELECT MIN(season), MAX(season), COUNT(DISTINCT season) FROM pitcher_model_scores")
print("pitcher_model_scores seasons:", c.fetchone())

# player_advanced_stats season range
c.execute("SELECT MIN(season), MAX(season), COUNT(DISTINCT season) FROM player_advanced_stats")
print("player_advanced_stats seasons:", c.fetchone())

# sample player_advanced_stats to see what real data looks like
c.execute("SELECT * FROM player_advanced_stats WHERE xfip IS NOT NULL LIMIT 3")
rows = c.fetchall()
c.execute("PRAGMA table_info(player_advanced_stats)")
cols = [r[1] for r in c.fetchall()]
for row in rows:
    print(dict(zip(cols, row)))

conn.close()
