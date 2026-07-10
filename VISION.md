# Zintoo Vision — Real-Time Fashion Image Classification & Visual Search

Adds image classification and visual similarity search to Zintoo, trained on the
Kaggle [Fashion Product Images dataset](https://www.kaggle.com/datasets/paramaggarwal/fashion-product-images-dataset).

---

## 1. What it does

Upload a garment photo → get back, in ~30–60 ms:

1. **Attributes**, with confidence: `articleType`, `baseColour`, `gender`, `masterCategory`
2. **Visually similar catalogue items**, ranked by cosine similarity
3. Optional **multimodal fusion** with a text query

A single backbone forward pass produces all of it.

## 2. Two decisions that differ from the brief

**No Flask/Django.** You already run FastAPI with auth, SSE and SQLite. Adding a
second framework means two servers, two auth systems, two deploys. The model is
served from the existing app. Same outcome, one process.

**No PyTorch at inference — ONNX Runtime.** `import torch` alone costs ~300 MB
RSS and ~800 MB on disk; that is precisely what forced the original deploy to
strip its ML deps and become a facade. We train in PyTorch and serve the export:

| | torch | onnxruntime |
|---|---|---|
| install size | ~800 MB | ~50 MB |
| RSS to import | ~300 MB | ~40 MB |
| MobileNetV3-S @224, 1 core | ~25 ms | ~15–30 ms |

Same weights, same numbers (parity-asserted at export), a fraction of the footprint.

## 3. Architecture

```
                       ┌──────────── OFFLINE (GPU, once) ────────────┐
 Kaggle dataset ──▶ prepare_data ──▶ train ──▶ export_onnx ──▶ build_index
   44k images         manifests     best.pt   classifier.onnx   embeddings.npy
                                              (+ labels.json)   (+ ids.json)
                       └──────────────────────┬──────────────────────┘
                                              │  ~12 MB of artifacts
                       ┌──────────────────────▼──────────── ONLINE (CPU) ─┐
   POST /recommend/image  ──▶ vision_preprocess ──▶ onnxruntime ──▶ heads ──▶ predictions
                                                          └──▶ embedding ──▶ visual_index
                                                                             (cosine, exact)
```

**Model.** Shared backbone (MobileNetV3-Small, or EfficientNet-B0 via
`ZINTOO_BACKBONE`) → 256-d L2-normalised embedding → one linear head per task.
Multi-head rather than four models: one forward pass instead of four, and the
auxiliary tasks regularise the trunk.

**Why the embedding matters.** Classification tells you "this is a Blue Casual
Shirt". Retailers want "show me what looks like this". The embedding gives you
that for free from the same forward pass.

**Why no FAISS.** 44k × 256 float16 = ~22 MB. Exact brute-force cosine is one
NumPy matmul, a few milliseconds. FAISS earns its complexity past ~1M items.

## 4. Two bug classes deliberately engineered out

**Train/serve skew.** Mean/std normalisation is *baked into the ONNX graph*
(`NormalizedModel` emits `Sub`/`Div` at the front). The server hands over a plain
`[0,1]` tensor and therefore **cannot** normalise incorrectly, because it never
normalises. `tests/test_vision.py` asserts the graph's arithmetic matches
`vision_preprocess.normalize_reference` (verified to 3e-7).

**Silent export drift.** `ml/export_onnx.py` refuses to write the artifact unless
PyTorch and ONNX Runtime agree to within 1e-4 across 5 random-input trials, and
asserts embeddings are unit-norm.

`ml/build_index.py` embeds the catalogue with **the exported ONNX and the
server's own preprocessing module**, so index vectors and query vectors provably
live in the same space.

## 5. Training pipeline

Needs a GPU. Kaggle's free T4 is plenty (~10 min/epoch, ~12 epochs).

```bash
pip install -r requirements-train.txt
export ZINTOO_DATASET_DIR=/path/to/fashion-dataset   # contains styles.csv + images/

python -m ml.prepare_data     # clean, validate, filter long tail, stratified split
python -m ml.train            # multi-head, AMP, class-weighted CE, early stop
python -m ml.export_onnx      # export + PARITY ASSERTION
python -m ml.build_index      # catalogue embeddings
python -m ml.evaluate         # top-1/top-5, macro-F1, worst classes, latency
python -m ml.import_catalog --inventory   # load the real 44k catalogue into SQLite
```

Then copy `models_artifacts/` (≈12 MB: `.onnx`, `labels.json`,
`catalog_embeddings.npy`, `catalog_ids.json`) into the repo and deploy.

**Dataset realities the pipeline handles.** `styles.csv` has malformed rows
(unescaped commas in `productDisplayName`) → skipped. Some ids have no image file
→ dropped before training, not mid-epoch. `articleType` has ~140 classes with a
savage long tail → classes below 50 samples are dropped and the retained coverage
is reported. NaNs in `baseColour`/`season`/`usage` → handled.

**Model selection is on macro-F1, not accuracy.** On a long-tailed catalogue,
accuracy is dominated by Tshirts and tells you nothing about the classes you care
about.

## 6. Serving

```bash
pip install -r requirements.txt
export ZINTOO_VISION_ENABLED=true
python -m uvicorn app.main:app --reload --port 8000
```

Open the **Discovery** page and drop an image on the upload zone. You get
attribute chips with confidences, per-request latency, and a grid of lookalikes.

```bash
curl -s localhost:8000/vision/health | python3 -m json.tool
curl -s -X POST "localhost:8000/recommend/image?top_k=5" -F "file=@shirt.jpg"
curl -s -X POST "localhost:8000/recommend/image?top_k=5&text_query=formal&alpha=0.7" -F "file=@shirt.jpg"
```

| Endpoint | Behaviour |
|---|---|
| `POST /recommend/image` | Classify + visual search. `400` on bad image, `503` if vision off/model missing |
| `GET /vision/health` | Artifact presence, task/class counts, embedding dim, test metrics |

Every classification also publishes a `vision.classify` event to the live SSE
feed, so the dashboard shows real inference traffic.

## 7. Environment variables

| Var | Default | Purpose |
|---|---|---|
| `ZINTOO_VISION_ENABLED` | `false` | Master switch. Off = `503`, and ORT is never imported |
| `ZINTOO_VISION_PRELOAD` | `false` | Load+warm at boot (slower start, no first-request spike) |
| `ZINTOO_VISION_THREADS` | `1` | ORT intra-op threads. **Keep at 1** on shared vCPU |
| `ZINTOO_MODELS_DIR` | `./models_artifacts` | Where the `.onnx` / `.npy` / `.json` live |
| `ZINTOO_DATASET_DIR` | `./data/fashion-dataset` | Training only |
| `ZINTOO_BACKBONE` | `mobilenet_v3_small` | or `efficientnet_b0` |

`ZINTOO_VISION_ENABLED` defaults to **off** so an existing deploy cannot OOM on
upgrade. Turn it on deliberately.

## 8. Performance, honestly

Measured in this environment (`tests/test_vision.py`, 1 CPU thread):

| Stage | Measured |
|---|---|
| Preprocess (decode → resize → crop → tensor) | **9.9 ms** — real |
| ORT inference | 0.44 ms — **stub model, not representative** |
| Visual search (300 items) | <1 ms |
| Peak RSS (ORT + NumPy + Pillow + app) | **92 MB** |

The stub model is a single matmul. **A real MobileNetV3-Small is ~15–30 ms per
image on one core**, so expect ~30–45 ms end-to-end. Run `python -m ml.evaluate`
after training for the true p50/p95/p99 on your hardware — I will not quote
accuracy or latency numbers for a model I have not trained.

**Memory.** Vision on adds roughly 150–250 MB RSS (ORT arenas + the model + a
22 MB mmapped index). On a 512 MB instance that is *tight but plausible*
alongside the ~90 MB base app. **Measure before enabling in production.** If it
OOMs: keep `ZINTOO_VISION_ENABLED=false` on the web service and run vision as a
separate, larger service.

## 9. Deploying on Render

Free tier is genuinely marginal for vision. Two options:

**A — text-only web service (safe).** Leave `ZINTOO_VISION_ENABLED=false`.
Comment out `onnxruntime`/`Pillow`/`numpy` in `requirements.txt`. `/recommend/image`
returns a clean `503`; everything else works.

**B — vision enabled (needs headroom).** Move to an instance with ≥1 GB RAM, set
`ZINTOO_VISION_ENABLED=true` and `ZINTOO_VISION_PRELOAD=true`, and commit
`models_artifacts/` (~12 MB — fine for git; don't commit the 25 GB dataset).

Keep `--workers 1`. Each worker loads its own ORT session and its own copy of the
arenas; two workers doubles the memory for no throughput gain on a fractional vCPU.

For real product images in the UI, serve `images/` from object storage (S3/R2 +
CDN), not from the web dyno. 44k JPEGs do not belong in your git repo or on an
ephemeral disk.

## 10. Testing

```bash
PYTHONPATH=. ZINTOO_WEATHER_ENABLED=false python tests/test_vision.py
```

Runs the full serving path against a **real onnxruntime session** — no trained
model, no torch, no network required (`tests/_onnx_builder.py` synthesises a
valid ONNX file). It asserts: preprocessing shape/range across aspect ratios and
colour modes; hostile input (empty, garbage, truncated, oversized) → `400` not
`500`; multi-head outputs sorted with valid probabilities; embeddings unit-norm;
softmax numerically stable at large logits; exact self-retrieval from the visual
index; embedding-dim mismatch rejected.

## 11. Known limitations

- **I have not trained this model.** No GPU, no torch, and no way to fetch a
  25 GB dataset in my environment. The training code is written and reviewed; the
  *serving* path is tested end-to-end against real ONNX Runtime. Report accuracy
  from `ml/evaluate.py`, not from me.
- **Dataset bias.** These are clean studio product shots on white backgrounds. On
  real-world customer photos (bad lighting, cluttered background, worn garments)
  accuracy will drop substantially. Fine-tune on in-domain photos before promising
  clients a number.
- **Dropped classes.** Article types with <50 samples are excluded. The model
  cannot predict them and will confidently pick a neighbour instead. Show the
  confidence score in client UIs and threshold it.
- **Exact search doesn't scale forever.** Past ~1M catalogue items, swap
  `visual_index` for an ANN index (hnswlib/FAISS).
- **The small dataset variant (60×80 px) is a poor fit** for 224×224 training.
  Use the full-resolution dataset.
