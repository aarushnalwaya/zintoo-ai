"""Guards the silent failure mode: the visual index stores Kaggle product ids,
so the products table MUST contain those ids or the similar-products grid comes
back empty with no error. Run after `ml.build_index`.

    PYTHONPATH=. python tests/test_catalog_linkage.py
"""
import os, tempfile, json
from pathlib import Path
import numpy as np
d=Path(tempfile.mkdtemp())
os.environ.update(ZINTOO_DATA_DIR=str(d), ZINTOO_SECRET_KEY="t", ZINTOO_MODELS_DIR=str(d/"m"),
                  ZINTOO_WEATHER_ENABLED="false", ZINTOO_VISION_ENABLED="true")
(d/"m").mkdir(parents=True)
from app import seed, db, visual_index

seed.seed_if_empty()
print("synthetic seed product_ids:", [r["product_id"] for r in db.query("SELECT product_id FROM products LIMIT 5")])

# Simulate what build_index.py produces: REAL Kaggle ids (e.g. 15970, 39386...)
kaggle_ids = [15970, 39386, 59263, 21379, 53759]
M = np.random.default_rng(0).normal(size=(len(kaggle_ids),32)).astype(np.float32)
M /= np.linalg.norm(M,axis=1,keepdims=True)
np.save(d/"m"/"catalog_embeddings.npy", M.astype(np.float16))
(d/"m"/"catalog_ids.json").write_text(json.dumps(kaggle_ids))

print("\n### CASE A: index built from Kaggle, but catalog NOT imported ###")
hits = visual_index.index.search(M[0], top_k=5)
print(f"  visual search returned {len(hits)} results")
if len(hits)==0:
    print("  ⚠️  CONFIRMED: embeddings reference Kaggle ids (15970...) that don't exist")
    print("     in the products table (1..400). _hydrate() drops them all -> EMPTY grid.")
    print("     => `python -m ml.import_catalog` is MANDATORY, not optional.")

print("\n### CASE B: after importing the real catalog ###")
with db.transaction() as conn:
    conn.execute("DELETE FROM products")
    for pid in kaggle_ids:
        conn.execute("INSERT INTO products(product_id,name,master_category,sub_category,"
                     "article_type,color,gender,season,usage,description) VALUES(?,?,?,?,?,?,?,?,?,?)",
                     (pid,f"Real Product {pid}","Apparel","Topwear","Shirts","Blue","Men","Summer","Casual","desc"))
visual_index.index._emb=None   # force reload
hits = visual_index.index.search(M[0], top_k=5)
print(f"  visual search returned {len(hits)} results; top = id {hits[0]['product_id']} "
      f"{hits[0]['name']!r} score {hits[0]['similarity_score']}")
assert len(hits)==5 and hits[0]["product_id"]==kaggle_ids[0]
print("  ✅ ids line up, rows hydrate, similarity ordering correct")
