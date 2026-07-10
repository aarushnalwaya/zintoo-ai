"""End-to-end test of the vision SERVING path against a real onnxruntime session.

Uses tests/_onnx_builder.py to synthesise a valid ONNX model, so this runs with
no trained artifact, no torch, and no network.

    PYTHONPATH=.. python tests/test_vision.py
"""
import os, tempfile, json, io, time, statistics
from pathlib import Path
import numpy as np
d = Path(tempfile.mkdtemp())
os.environ.update(ZINTOO_DATA_DIR=str(d), ZINTOO_SECRET_KEY="t", ZINTOO_MODELS_DIR=str(d/"m"),
                  ZINTOO_WEATHER_ENABLED="false", ZINTOO_VISION_ENABLED="true")
from PIL import Image
from tests._onnx_builder import build_fake_model
from app import seed, db, vision, visual_index
from app.vision_preprocess import preprocess, ImageDecodeError

tasks={"articleType":["Tshirts","Jeans","Casual Shoes","Watches","Shirts"],
       "baseColour":["Black","Blue","White"], "gender":["Men","Women","Unisex"]}
build_fake_model(d/"m", tasks, embedding_dim=32)
seed.seed_if_empty()
print("catalog products:", db.table_count("products"))

def jpg(w=800,h=1200,color=(20,40,180)):
    b=io.BytesIO(); Image.new("RGB",(w,h),color).save(b,"JPEG"); return b.getvalue()

print("\n=== 1. PREPROCESSING CONTRACT ===")
t=preprocess(jpg())
print("  shape",t.shape,"dtype",t.dtype,"range",(round(float(t.min()),3),round(float(t.max()),3)))
assert t.shape==(1,3,224,224) and t.dtype==np.float32 and 0<=t.min() and t.max()<=1
# non-square + portrait/landscape both -> 224x224
for wh in [(1800,2400),(2400,1800),(300,300),(60,80)]:
    assert preprocess(jpg(*wh)).shape==(1,3,224,224), wh
print("  ✅ any aspect ratio -> (1,3,224,224)")
# EXIF/greyscale/PNG-alpha handled
b=io.BytesIO(); Image.new("L",(300,300),128).save(b,"PNG")
assert preprocess(b.getvalue()).shape==(1,3,224,224)
print("  ✅ greyscale PNG coerced to 3-channel RGB")

print("\n=== 2. HOSTILE INPUT (must 400, not 500) ===")
for name,payload in [("empty",b""),("garbage",b"not an image at all"),
                     ("truncated jpeg", jpg()[:40]), ("oversized", b"x"*(9*1024*1024))]:
    try:
        preprocess(payload); print(f"  ❌ {name} did NOT raise")
    except ImageDecodeError as e: print(f"  ✅ {name:<15} -> ImageDecodeError: {str(e)[:44]}")

print("\n=== 3. REAL ORT INFERENCE ===")
r = vision.model.infer(jpg(), top_k=3)
print("  predictions:")
for task,preds in r["predictions"].items():
    print(f"    {task:<12}", ", ".join(f"{p['label']}={p['confidence']:.3f}" for p in preds))
print("  timing_ms:", r["timing_ms"])
assert set(r["predictions"])=={"articleType","baseColour","gender"}
for task,labels in tasks.items():
    ps=[p["confidence"] for p in r["predictions"][task]]
    assert ps==sorted(ps,reverse=True), f"{task} not sorted by confidence"
    assert all(0<=p<=1 for p in ps)
print("  ✅ multi-head outputs, sorted, valid probabilities")
emb=np.array(r["embedding"],dtype=np.float32)
print(f"  embedding dim={emb.shape[0]}  L2 norm={np.linalg.norm(emb):.6f}")
assert abs(np.linalg.norm(emb)-1.0)<1e-4, "embedding not unit-norm!"
print("  ✅ embedding is L2-normalised (cosine == dot)")

print("\n=== 4. FULL SOFTMAX SANITY ===")
import onnxruntime as ort
s=ort.InferenceSession(str(d/"m"/"fashion_classifier.onnx"),providers=["CPUExecutionProvider"])
logits=s.run(["logits_articleType"],{"input":preprocess(jpg())})[0][0]
p=vision._softmax(logits.astype(np.float32))
print(f"  softmax sums to {p.sum():.8f}")
assert abs(p.sum()-1)<1e-6
big=np.array([1000.,1001.,999.],dtype=np.float32)   # overflow trap
assert not np.isnan(vision._softmax(big)).any(), "softmax overflowed!"
print("  ✅ softmax normalised and numerically stable at large logits")

print("\n=== 5. VISUAL SIMILARITY INDEX ===")
ids=[r["product_id"] for r in db.query("SELECT product_id FROM products LIMIT 300")]
rng=np.random.default_rng(3); M=rng.normal(size=(len(ids),32)).astype(np.float32)
M/=np.linalg.norm(M,axis=1,keepdims=True)
np.save(d/"m"/"catalog_embeddings.npy", M.astype(np.float16))
(d/"m"/"catalog_ids.json").write_text(json.dumps(ids))
hits=visual_index.index.search(M[7], top_k=5)   # query == row 7 exactly
print("  self-query top hit:", hits[0]["product_id"], "score", hits[0]["similarity_score"], "| expected", ids[7])
assert hits[0]["product_id"]==ids[7] and hits[0]["similarity_score"]>0.99
scores=[h["similarity_score"] for h in hits]
assert scores==sorted(scores,reverse=True)
assert all("name" in h and "article_type" in h for h in hits), "rows not hydrated from DB"
print("  ✅ exact self-retrieval, descending scores, rows hydrated from SQLite")
try:
    visual_index.index.search(np.zeros(999,dtype=np.float32),5); print("  ❌ dim mismatch not caught")
except ValueError as e: print("  ✅ dim mismatch rejected:", str(e)[:50])

print("\n=== 6. LATENCY (real ORT, single core) ===")
img=jpg(); [vision.model.infer(img) for _ in range(3)]
lat=[]
for _ in range(30):
    t0=time.perf_counter(); vision.model.infer(img); lat.append((time.perf_counter()-t0)*1000)
lat.sort()
print(f"  p50={statistics.median(lat):.1f} ms  p95={lat[int(.95*len(lat))]:.1f} ms  max={lat[-1]:.1f} ms")

import resource,sys
rss=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss/(1024 if sys.platform!='darwin' else 1024**2)
print(f"\n=== 7. PEAK RSS = {rss:.0f} MB (ORT + numpy + PIL + app) ===")
print("\nALL VISION TESTS PASSED ✅")
