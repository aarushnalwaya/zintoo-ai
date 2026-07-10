"""Tests the stdlib (pandas-free) catalogue importer: malformed-row recovery,
column alignment, defaults, dedupe, transactional import, idempotency.

    PYTHONPATH=. python tests/test_import_catalog.py
"""
import os, tempfile, sys, importlib
from pathlib import Path
d=Path(tempfile.mkdtemp()); ds=d/"fashion-data"; ds.mkdir()
os.environ.update(ZINTOO_DATA_DIR=str(d/"rt"), ZINTOO_SECRET_KEY="t",
                  ZINTOO_WEATHER_ENABLED="false", ZINTOO_DATASET_DIR=str(ds))

csv_text = """id,gender,masterCategory,subCategory,articleType,baseColour,season,year,usage,productDisplayName
15970,Men,Apparel,Topwear,Shirts,Navy Blue,Fall,2011,Casual,Turtle Check Men Navy Blue Shirt
39386,Men,Apparel,Bottomwear,Jeans,Blue,Summer,2012,Casual,Peter England Men Party Blue Jeans
59263,Women,Accessories,Watches,Watches,Silver,Winter,2016,Casual,Titan Women Silver Watch
21379,Men,Apparel,Bottomwear,Track Pants,Black,Fall,2011,Casual,Manchester United Men Solid Black Track Pants
53759,Men,Apparel,Topwear,Tshirts,Grey,Summer,2012,Casual,Puma Men Grey T-shirt
99001,Men,Apparel,Topwear,Shirts,Navy Blue,Fall,2011,Casual,Turtle Check Men, Navy Blue, Shirt
99002,Women,Apparel,Topwear,Tops,,Summer,2012,Casual,
99003,Men,,,,,,,,Broken Row
99004,Men,Apparel
15970,Men,Apparel,Topwear,Shirts,Red,Fall,2011,Casual,DUPLICATE ID should be ignored
"""
(ds/"styles.csv").write_text(csv_text, encoding="utf-8")

sys.path.insert(0,os.getcwd())
from ml import import_catalog
from app import db

print("=== stdlib parser (no pandas) ===")
rows = import_catalog.load_catalog()
byid = {r["id"]: r for r in rows}
print()
assert 99001 in byid, "malformed row with embedded commas was NOT recovered"
print(f"  ✅ recovered malformed row 99001, name = {byid[99001]['name']!r}")
assert byid[99001]["name"] == "Turtle Check Men, Navy Blue, Shirt"
assert byid[99001]["articleType"] == "Shirts" and byid[99001]["baseColour"] == "Navy Blue"
print("  ✅ its columns are still aligned (articleType/baseColour correct)")
assert byid[99002]["baseColour"] == "Unknown", "empty colour should default"
assert byid[99002]["name"] == "Unknown Tops", f"empty name should synthesise: {byid[99002]['name']}"
print(f"  ✅ empty fields defaulted: name={byid[99002]['name']!r}")
assert 99003 not in byid and 99004 not in byid
print("  ✅ genuinely unusable rows skipped (99003 blank article, 99004 short)")
assert byid[15970]["name"].startswith("Turtle"), "duplicate id must keep FIRST occurrence"
print("  ✅ duplicate id 15970 deduped (kept first)")

try:
    import pandas as pd
    print("\n=== compare vs pandas on_bad_lines='skip' ===")
    pdf = pd.read_csv(ds/"styles.csv", on_bad_lines="skip", engine="python")
    print(f"  pandas kept {len(pdf)} rows | stdlib kept {len(rows)} rows")
    assert 99001 not in set(pdf['id']), "sanity: pandas should drop the malformed row"
    print("  ✅ stdlib recovers a row pandas silently discards")
except ImportError:
    print("\n(pandas not installed — skipping comparison; not needed to serve)")

print("\n=== full import into SQLite ===")
db.init_db()
n = import_catalog.import_products(rows)
m = import_catalog.regenerate_inventory(rows, top_n=3)
print(f"  imported {n} products, {m} inventory rows")
got = db.query_one("SELECT name, article_type FROM products WHERE product_id=99001")
print(f"  DB row 99001: {got['name']!r} / {got['article_type']}")
assert got["name"] == "Turtle Check Men, Navy Blue, Shirt"
assert db.table_count("products") == n
print("  ✅ committed transactionally, comma-containing name survived SQL round-trip")

print("\n=== idempotency: re-import doesn't duplicate ===")
import_catalog.import_products(rows)
assert db.table_count("products") == n
print(f"  ✅ still {n} products after re-import")
print("\nALL IMPORT TESTS PASSED ✅")
