# Training Zintoo Vision on Google Colab (free T4)

Same result as the Kaggle path, different hazards. **Total ~2.5 hours**, free.

> **The one thing that ruins Colab runs:** free runtimes disconnect (idle ~90 min,
> hard cap ~12 h, and sometimes just because). Steps 2 and 7 make that survivable
> instead of catastrophic. Don't skip them.

---

## Step 1 — Notebook + GPU

1. [colab.research.google.com](https://colab.research.google.com) → **New notebook**
2. **Runtime → Change runtime type → Hardware accelerator: `T4 GPU`** → Save

Confirm you actually got one:

```python
!nvidia-smi
```

You should see `Tesla T4` and ~15 GB memory. If it says "cannot find GPU", the
free pool is exhausted — wait and retry.

Check your disk while you're there (you need ~55 GB free for the 25 GB dataset
plus its unzipped copy):

```python
!df -h /content | tail -1
```

---

## Step 2 — Mount Drive (this is your insurance policy)

Checkpoints go to Drive so a dead runtime doesn't cost you two hours.

```python
from google.colab import drive
drive.mount('/content/drive')

!mkdir -p "/content/drive/MyDrive/zintoo/models_artifacts"
```

**Images stay on local disk** (`/content`), never Drive — writing 44k small files
to Drive is agonisingly slow. Only checkpoints and the four final artifacts go there.

---

## Step 3 — Get the code

```python
!git clone https://github.com/aarushnalwaya/zintoo-ai.git /content/zintoo
%cd /content/zintoo
```

> If your repo root contains a folder named `zintoo ai` (with a space):
> `%cd "/content/zintoo/zintoo ai"`
>
> Not pushed to GitHub yet? Upload the zip instead:
> ```python
> from google.colab import files; files.upload()      # pick zintoo-ai-vision.zip
> !unzip -q zintoo-ai-vision.zip -d /content/zintoo
> %cd "/content/zintoo/zintoo ai"
> ```

---

## Step 4 — Install (don't use requirements-train.txt here)

Colab already ships `torch`, `torchvision`, `pandas`, and `scikit-learn` with
working CUDA. Installing the requirements file risks pip **downgrading torch** and
breaking GPU support. Install only what's genuinely missing:

```python
!pip install -q onnx onnxruntime
import torch; print(torch.__version__, "| CUDA:", torch.cuda.is_available())
```

`CUDA: True` or stop here and fix Step 1.

---

## Step 5 — Download the dataset (~15 min)

Get your Kaggle API token first: kaggle.com → your avatar → **Settings** →
**API** → **Create New Token** → downloads `kaggle.json`.

```python
from google.colab import files
files.upload()                      # select kaggle.json

!mkdir -p ~/.kaggle && cp kaggle.json ~/.kaggle/ && chmod 600 ~/.kaggle/kaggle.json
!pip install -q kaggle

!kaggle datasets download -d paramaggarwal/fashion-product-images-dataset -p /content
!unzip -q /content/fashion-product-images-dataset.zip -d /content/fashion
!rm /content/fashion-product-images-dataset.zip     # reclaim 25 GB — you need it
```

Now **find the real path** (it's nested, and guessing wastes an hour):

```python
!find /content/fashion -maxdepth 3 -name "styles.csv"
```

Whatever directory holds **both** `styles.csv` and `images/` is your dataset dir.

---

## Step 6 — Configure

```python
import os
os.environ["ZINTOO_DATASET_DIR"] = "/content/fashion/fashion-dataset"   # ← from Step 5
os.environ["ZINTOO_MODELS_DIR"]  = "/content/drive/MyDrive/zintoo/models_artifacts"  # Drive!
os.environ["ZINTOO_RESIZE_OUT"]  = "/content/fashion-resized"           # local disk, fast
os.environ["ZINTOO_DATA_DIR"]    = "/content/runtime"
os.environ["ZINTOO_NUM_WORKERS"] = "2"     # Colab gives ~2 usable cores
os.environ["ZINTOO_RESUME"]      = "true"  # ← survive disconnects
```

`ZINTOO_MODELS_DIR` on Drive is the whole trick: `last.pt` is written there after
**every epoch**, so a dead runtime costs you at most one epoch.

---

## Step 7 — Preflight (30 seconds, saves hours)

```python
!python -m ml.doctor
```

Verifies the path, that `styles.csv` parses, how many ids lack images, and the
resolution. If the path is wrong it suggests the right one. **Fix red before continuing.**

---

## Step 8 — Pre-resize ⚡ (~15 min, do not skip)

The dataset ships 1800×2400 JPEGs. Decoding one costs ~50 ms; with 44k images and
Colab's 2 CPU cores, **your T4 sits idle waiting for JPEG decode** — ~30 min/epoch
instead of ~3.

```python
!python -m ml.resize_cache
os.environ["ZINTOO_DATASET_DIR"] = "/content/fashion-resized"   # point training at the cache
```

Measured: **84× smaller files**, shorter side 256 px. Pays for itself before epoch one.

---

## Step 9 — Prepare data (~2 min)

```python
!python -m ml.prepare_data
```

Skips the malformed CSV rows, drops ids with no image, prunes `articleType`
classes under 50 samples, writes a stratified split. It prints exactly what it dropped.

---

## Step 10 — Smoke-test first (10 min) — seriously

Get a deliberately bad model end-to-end before committing two hours:

```python
os.environ["ZINTOO_EPOCHS"] = "1"
!python -m ml.train && python -m ml.export_onnx
```

If that survives export (it asserts torch↔ONNX parity), your pipeline is sound.
Now do it for real.

---

## Step 11 — Train (~1.5–2 hours) ☕

```python
os.environ["ZINTOO_EPOCHS"] = "12"
!python -m ml.train
```

Watch **`val_macroF1`**, not accuracy — on a long-tailed catalogue accuracy is
dominated by T-shirts and tells you nothing.

**Keep the tab open and interact occasionally.** Colab kills idle sessions.

### If it disconnects

Reconnect, re-run Steps 3, 4, 6 (Drive stays mounted; the dataset may need
re-downloading), then just run `!python -m ml.train` again. With
`ZINTOO_RESUME=true` it prints:

```
▶ resumed from last.pt at epoch 7 (best macro-F1 so far 0.7412)
```

and carries on. It restores the optimizer, LR schedule, and AMP scaler — not just
the weights, which is what makes the resume actually correct rather than
approximately correct.

---

## Step 12 — Export, index, evaluate (~15 min)

```python
!python -m ml.export_onnx    # refuses to write unless torch↔ONNX agree to 1e-4
!python -m ml.build_index    # embeds all 44k images through the exported ONNX
!python -m ml.evaluate       # top-1/top-5, macro-F1, 10 weakest classes, real latency
```

Step 12's `evaluate` output is where your accuracy number comes from. I have not
trained this model, so I have quoted none.

---

## Step 13 — Collect the artifacts

They're already on Drive. Confirm:

```python
!ls -lh "/content/drive/MyDrive/zintoo/models_artifacts/"
```

Download these **four** files to your PC (`models_artifacts\` in your project):

- `fashion_classifier.onnx`
- `labels.json`
- `catalog_embeddings.npy`
- `catalog_ids.json`

You do **not** need `best.pt` / `last.pt` to serve — they're PyTorch checkpoints.

Also grab `styles.csv` (a few MB) — you need it locally for Step 14:

```python
!cp /content/fashion/fashion-dataset/styles.csv "/content/drive/MyDrive/zintoo/"
```

---

## Step 14 — Wire it up locally (Windows) — **mandatory**

```cmd
cd "C:\Users\aarus\Downloads\zintoo-ai-vision\zintoo ai"
.venv\Scripts\activate

set ZINTOO_DATASET_DIR=C:\path\to\folder\containing\styles.csv
python -m ml.import_catalog --inventory
python -m ml.doctor
```

### Why this step is not optional

`build_index` stores **Kaggle product ids** (15970, 39386…). Your database holds
the synthetic seed (ids 1–400). Skip `import_catalog` and classification works
while the similar-products grid comes back **empty, with no error**. I verified
this failure mode explicitly — `tests/test_catalog_linkage.py` pins it.

`doctor` must print `✅ ids line up — visual search will hydrate real products`.

---

## Step 15 — Turn vision on

```cmd
set ZINTOO_VISION_ENABLED=true
set ZINTOO_VISION_PRELOAD=true
python -m uvicorn app.main:app --reload --port 8000
```

```cmd
curl http://localhost:8000/vision/health
```

Open **AI Discovery**, drop a shirt photo. You get attribute chips with
confidences, per-request latency, and a grid of visually similar products.

> `set` only lasts for that terminal. Put `ZINTOO_VISION_ENABLED=true` in a `.env`
> file to make it permanent.

---

## Colab vs Kaggle — which to use

| | Colab (T4) | Kaggle (T4 ×2 / P100) |
|---|---|---|
| Dataset access | download 25 GB yourself (~15 min) | one click, pre-mounted |
| Disk | ~78 GB (tight after unzip) | ~73 GB, dataset doesn't count |
| Disconnects | frequent — needs Drive + resume | rarer, 12 h sessions |
| Setup effort | higher (kaggle.json, unzip, Drive) | lower |

**Kaggle is genuinely the easier path for this dataset** — it's already mounted,
no download, no unzip, no 25 GB of disk pressure. Use Colab if you prefer it or
Kaggle's GPU quota is exhausted. Both produce identical artifacts.

---

## Troubleshooting

| Symptom | Fix |
|---|---|
| `CUDA: False` | Runtime → Change runtime type → T4 GPU |
| `No space left on device` | You forgot `!rm` on the 25 GB zip (Step 5) |
| Epochs take ~30 min | Skipped Step 8 (resize cache) |
| Session died, lost everything | `ZINTOO_MODELS_DIR` wasn't on Drive. Step 2/6 |
| Resume starts from epoch 1 | `ZINTOO_RESUME` not set, or `last.pt` not on Drive |
| torch downgraded / CUDA broke | You ran `pip install -r requirements-train.txt`. Step 4 |
| Predictions work, **zero similar products** | `python -m ml.import_catalog --inventory` (Step 14) |
| `PARITY FAILURE` at export | Don't ship it. Re-export; if it persists, retrain |
| `No module named 'app'` | `cd` into the folder containing `app/` |

---

## Before you show clients

1. **This dataset is clean studio shots on white backgrounds.** On real customer
   phone photos — poor lighting, cluttered rooms, garments being worn — accuracy
   drops substantially. Fine-tune on in-domain images before quoting a number.
2. **Long-tail classes are dropped** (<50 samples). The model can't predict them
   and will confidently pick a neighbour. Show the confidence score and threshold it.
3. **Take your accuracy from `ml/evaluate.py`.** Its "ten weakest classes" output
   is the honest picture of where the model fails.
