import os, tempfile, sys
from pathlib import Path
d=Path(tempfile.mkdtemp()); ds=d/"fashion-data"; ds.mkdir()
os.environ.update(ZINTOO_DATA_DIR=str(d/"rt"), ZINTOO_SECRET_KEY="t",
                  ZINTOO_WEATHER_ENABLED="false", ZINTOO_DATASET_DIR=str(ds))
sys.path.insert(0,os.getcwd())

# a mini "real" catalog with Kaggle-style ids
hdr="id,gender,masterCategory,subCategory,articleType,baseColour,season,year,usage,productDisplayName"
rows=[hdr]
for i,pid in enumerate([15970,39386,59263,21379,53759,10005,42819,6534]):
    art=["Shirts","Jeans","Watches","Tshirts"][i%4]
    rows.append(f"{pid},Men,Apparel,Topwear,{art},Blue,Summer,2012,Casual,Real Product {pid}")
(ds/"styles.csv").write_text("\n".join(rows))

from ml import import_catalog
from app import db, seed

print("### YOUR EXACT SEQUENCE ###")
print("1) python -m ml.import_catalog --inventory")
db.init_db()
r = import_catalog.load_catalog()
n = import_catalog.import_products(r)
m = import_catalog.regenerate_inventory(r, top_n=4)
db.set_meta("catalog_source","kaggle:fashion-product-images-dataset")
real_ids = {x["id"] for x in r}
print(f"   -> {n} real products, {m} inventory rows")
inv_before = [x["sku"] for x in db.query("SELECT DISTINCT sku FROM inventory")]
print(f"   real SKUs: {inv_before}")

print("\n2) start the app  (seed_if_empty runs)")
seed.seed_if_empty()
after = db.table_count("products")
print(f"   -> products now: {after}")

synthetic = db.query("SELECT product_id,name FROM products WHERE product_id <= 400")
print(f"   synthetic (id<=400) rows injected: {len(synthetic)}")
assert after == n, f"CONTAMINATION: {after} != {n}"
assert len(synthetic)==0, "synthetic products leaked into the real catalogue"
print("   ✅ real catalogue untouched, zero synthetic products")

inv_after = [x["sku"] for x in db.query("SELECT DISTINCT sku FROM inventory")]
assert inv_after == inv_before, f"inventory overwritten! {inv_after}"
print(f"   ✅ real inventory preserved: {inv_after}")

dem = db.query("SELECT DISTINCT sku FROM demand_history")
dem_skus = sorted(x["sku"] for x in dem)
print(f"   ✅ demand history generated for REAL skus: {dem_skus[:4]}")
assert set(dem_skus) <= set(inv_before), "demand references SKUs not in inventory!"

print("\n3) restart the app again (idempotency)")
seed.seed_if_empty()
assert db.table_count("products")==n and [x["sku"] for x in db.query("SELECT DISTINCT sku FROM inventory")]==inv_before
print(f"   ✅ still {n} products, inventory stable")

print("\n### FRESH INSTALL (no import) still seeds normally ###")
d2=Path(tempfile.mkdtemp()); os.environ["ZINTOO_DATA_DIR"]=str(d2)
import importlib
from app import settings as S
importlib.reload(S); importlib.reload(db); importlib.reload(seed)
seed.seed_if_empty()
print(f"   products={db.table_count('products')} inventory={db.table_count('inventory')} demand={db.table_count('demand_history')}")
assert db.table_count("products")>0 and db.table_count("demand_history")>0
print("   ✅ synthetic demo still works out of the box")
print("\nALL CONTAMINATION TESTS PASSED ✅")
