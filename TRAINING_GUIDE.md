# Training Zintoo Vision — Step by Step

Goal: turn that orange "Visual search unavailable" banner into working image
classification + visual similarity search.

**Total time:** ~2.5 hours, mostly unattended. **Cost:** free (Kaggle GPU).

---

## Before you start: what that orange banner means

Nothing is broken. `/recommend/image` returned **HTTP 503** because
`ZINTOO_VISION_ENABLED=false` and there is no trained model yet. That is the
designed behaviour — the app tells you the truth instead of showing fake results.

At the end of this guide it becomes a real classifier.

---

## Step 0 — Get the code onto Kaggle

Kaggle needs your project files. Easiest is GitHub:

```cmd
cd "C:\Users\aarus\Downloads\zintoo-ai-vision\zintoo ai"
git init
git add .
git commit -m "Zintoo v2 + vision pipeline"
git remote add origin https://github.com/aarushnalwaya/zintoo-ai.git
git branch -M main
git push -u origin main --force
```

> If you'd rather not use git: zip the `zintoo ai` folder and upload it to Kaggle
> as a private **Dataset**, then `!cp -r /kaggle/input/<your-dataset>/* /kaggle/working/`.

---

## Step 1 — Create the Kaggle notebook

1. kaggle.com → **Create** → **New Notebook**
2. Right sidebar → **Add Input** → search `fashion-product-images-dataset`
   (by *paramaggarwal*) → **Add**
3. Right sidebar → **Session options**:
   - **Accelerator:** `GPU T4 x2` (or P100)
   - **Internet:** `On` ← **required** (pip + pretrained weights). Needs phone
     verification on your Kaggle account. Without this, Step 3 fails.

---

## Step 2 — Find the dataset (do not guess the path)

Kaggle often nests the folder one level deeper than you expect. In a cell:

```python
!find /kaggle/input -name "styles.csv" -maxdepth 4
!ls /kaggle/input/fashion-product-images-dataset/
```

Whatever directory contains **both** `styles.csv` and `images/` is your
`ZINTOO_DATASET_DIR`. It is usually:

```
/kaggle/input/fashion-product-images-dataset/fashion-dataset
```

---

## Step 3 — Set up

```python
!git clone https://github.com/aarushnalwaya/zintoo-ai.git /kaggle/working/zintoo
%cd /kaggle/working/zintoo
!pip install -q -r requirements-train.txt

import os
os.environ["ZINTOO_DATASET_DIR"] = "/kaggle/input/fashion-product-images-dataset/fashion-dataset"  # ← from Step 2
os.environ["ZINTOO_MODELS_DIR"]  = "/kaggle/working/models_artifacts"
os.environ["ZINTOO_DATA_DIR"]    = "/kaggle/working/runtime"
os.environ["ZINTOO_NUM_WORKERS"] = "2"     # Kaggle gives ~2 usable CPU cores
```

> If your repo folder is `zintoo ai` (with a space), `%cd "/kaggle/working/zintoo/zintoo ai"`.

---

## Step 4 — Preflight (30 seconds, saves hours)

```python
!python -m ml.doctor
```

This verifies the dataset path, that `styles.csv` parses, how many ids lack an
image, and image resolution. **Fix anything red before continuing.** If the path
is wrong it will even suggest the correct one.

---

## Step 5 — Pre-resize the images ⚡ (~15 min, do not skip)

The dataset ships 1800×2400 JPEGs. Decoding those is ~50 ms each; with 44k images
and 2 dataloader workers, **your GPU will sit idle waiting for JPEG decode** and
an epoch takes ~30 minutes instead of ~3.

```python
!python -m ml.resize_cache
os.environ["ZINTOO_DATASET_DIR"] = "/kaggle/input/.../fashion-dataset-resized"  # path it prints
```

Measured on sample images: **84× smaller files**, shorter side 256 px. This one
step pays for itself before the first epoch finishes.

> The cache is written to `/kaggle/working/`, which is capped at ~20 GB. The
> resized set is a few hundred MB — fine.

---

## Step 6 — Prepare the data (~2 min)

```python
!python -m ml.prepare_data
```

Cleans the malformed CSV rows, drops ids with no image, prunes `articleType`
classes below 50 samples (the dataset has ~140 classes with a brutal long tail),
and writes a stratified train/val/test split. It prints exactly what it dropped.

---

## Step 7 — Train (~1.5–2 hours) ☕

```python
!python -m ml.train
```

Watch the `val_macroF1` column, not accuracy. On a long-tailed catalogue,
accuracy is dominated by T-shirts and tells you nothing.

It early-stops, keeps the best checkpoint, and prints held-out test metrics.

**To smoke-test the whole pipeline in ~10 minutes first** (recommended):

```python
os.environ["ZINTOO_EPOCHS"] = "1"
!python -m ml.train
```

Get a bad-but-working model end to end, confirm Steps 8–11 work, *then* rerun
with 12 epochs.

---

## Step 8 — Export + verify (~1 min)

```python
!python -m ml.export_onnx
```

This **refuses to write the artifact** unless PyTorch and ONNX Runtime agree to
within 1e-4 on random inputs, and asserts the embeddings are unit-norm. If it
fails, do not ship — the exported model differs from the one you evaluated.

---

## Step 9 — Build the visual index (~10 min)

```python
!python -m ml.build_index
```

Embeds all ~44k catalogue images **through the exported ONNX**, so index vectors
and query vectors provably live in the same space.

---

## Step 10 — Evaluate (honest numbers)

```python
!python -m ml.evaluate
```

Prints top-1/top-5, macro-F1, your **ten weakest classes**, and real p50/p95/p99
latency. These are your numbers. I have not trained this model, so I have quoted
no accuracy figure anywhere — this command is where it comes from.

---

## Step 11 — Download the artifacts (~12–25 MB)

```python
!ls -lh /kaggle/working/models_artifacts/
```

From the Kaggle **Output** panel, download these four files:

- `fashion_classifier.onnx`
- `labels.json`
- `catalog_embeddings.npy`
- `catalog_ids.json`

You do **not** need `best.pt` (the PyTorch checkpoint) to serve.

---

## Step 12 — Wire it up locally (Windows)

Put the four files in `models_artifacts\` inside your project folder.

You also need `styles.csv` locally — download just that one file from the Kaggle
dataset (a few MB).

```cmd
cd "C:\Users\aarus\Downloads\zintoo-ai-vision\zintoo ai"
.venv\Scripts\activate

set ZINTOO_DATASET_DIR=C:\path\to\folder\containing\styles.csv
python -m ml.import_catalog --inventory
```

### ⚠️ Step 12 is mandatory, not optional

`build_index` stores **Kaggle product ids** (15970, 39386…). Your database
currently holds the synthetic seed (ids 1–400). If you skip `import_catalog`,
classification will work but the similar-products grid will come back **empty,
with no error message**. I verified this failure mode explicitly
(`tests/test_catalog_linkage.py`).

Confirm the linkage before you trust it:

```cmd
python -m ml.doctor
```

It must print `✅ ids line up — visual search will hydrate real products`.

---

## Step 13 — Turn vision on

```cmd
set ZINTOO_VISION_ENABLED=true
set ZINTOO_VISION_PRELOAD=true
python -m uvicorn app.main:app --reload --port 8000
```

Check the subsystem:

```cmd
curl http://localhost:8000/vision/health
```

Then open **AI Discovery** and drop a shirt photo on the upload zone. You should
see attribute chips with confidences, the per-request latency, and a grid of
visually similar products.

> To make it permanent, put `ZINTOO_VISION_ENABLED=true` in a `.env` file rather
> than `set`, which only lasts for that terminal session.

---

## Step 14 — Deploy for clients

Commit the four artifacts (~12–25 MB is fine for git; **never** commit the 25 GB
dataset — `.gitignore` already blocks it):

```cmd
git add models_artifacts/ && git commit -m "Add trained vision artifacts" && git push
```

On Render, set `ZINTOO_VISION_ENABLED=true`.

**Read `VISION.md` §9 first.** Vision adds ~150–250 MB RSS on top of the ~90 MB
base app. On the 512 MB free plan that is tight and may OOM. Use an instance with
≥1 GB RAM, keep `--workers 1`, and make sure the Start Command is
`uvicorn app.main:app` (not the old `api.main:app`).

---

## Troubleshooting

| Symptom | Cause | Fix |
|---|---|---|
| Orange "Visual search unavailable" | `ZINTOO_VISION_ENABLED` not set | Step 13 |
| `503 Vision model artifact not found` | Artifacts not in `models_artifacts/` | Step 11–12 |
| Predictions appear, **but zero similar products** | Catalogue linkage broken | `python -m ml.import_catalog --inventory` |
| `PARITY FAILURE` at export | Export drifted from checkpoint | Don't ship it. Re-export; if it persists, retrain |
| Epochs take ~30 min | Skipped the resize cache | Step 5 |
| `pip install` fails on Kaggle | Internet off | Session options → Internet: On |
| `No module named 'app'` | Wrong working directory | `cd` into the folder containing `app/` |
| App OOMs on Render | Vision on a 512 MB instance | Bigger instance, or `ZINTOO_VISION_ENABLED=false` |

---

## Set expectations with your clients

Three things I'd want you to know before promising anything:

1. **This dataset is clean studio product shots on white backgrounds.** On real
   customer phone photos — bad lighting, cluttered rooms, garments being worn —
   accuracy drops substantially. Fine-tune on in-domain photos before quoting a
   number.
2. **Long-tail classes are dropped** (<50 samples). The model cannot predict them
   and will confidently pick a neighbour instead. Show the confidence score in
   client-facing UIs and set a threshold below which you say "not sure".
3. **Report accuracy from `ml/evaluate.py`, not from marketing.** Its "ten weakest
   classes" output is the honest picture of where the model fails.
