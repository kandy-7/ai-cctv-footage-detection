import sqlite3
import os

dbs = [
    os.path.abspath("backend/cctv_events.db"),
    os.path.abspath("cctv_events.db")
]

for db_path in dbs:
    if not os.path.exists(db_path):
        continue
    print("=" * 60)
    print(f"DATABASE: {db_path} ({os.path.getsize(db_path)} bytes)")
    print("=" * 60)
    conn = sqlite3.connect(db_path)
    cur = conn.cursor()
    cur.execute("SELECT name FROM sqlite_master WHERE type='table';")
    tables = [r[0] for r in cur.fetchall()]
    print("Tables:", tables)
    for table in tables:
        print(f"\n--- Table: {table} ---")
        cur.execute(f"PRAGMA table_info({table});")
        cols = cur.fetchall()
        for c in cols:
            # cid, name, type, notnull, dflt_value, pk
            print(f"  Col #{c[0]}: {c[1]} ({c[2]}) | NOT NULL: {c[3]} | DEFAULT: {c[4]} | PK: {c[5]}")
        cur.execute(f"SELECT count(*) FROM {table};")
        count = cur.fetchone()[0]
        print(f"  Total Rows: {count}")
        if count > 0:
            cur.execute(f"SELECT * FROM {table} LIMIT 2;")
            print("  Sample rows:", cur.fetchall())
    conn.close()
