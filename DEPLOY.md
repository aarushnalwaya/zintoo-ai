# Deploying Zintoo AI so other people can use it

## ⛔ Read this first

Do **not** deploy the model you have now. From your `ml.evaluate` output:

| Task | Your 12-epoch model | Your 1-epoch smoke test |
|---|---|---|
| `masterCategory` top-1 | **0.383** | 0.986 |
| `gender` top-1 | **0.322** | 0.706 |
| `baseColour` top-1 | **0.155** | 0.224 |

The 12-epoch model is worse than the throwaway. `masterCategory` at 0.38 is
*below the majority-class baseline* — always guessing "Apparel" would score
higher. That is a collapsed model, not a weak one. It classified a Nike cap as
"Ties / Women / Free Items".

**Retrain before deploying.** Likely cause: you resumed after the scheduler
crash, and the LR-schedule rebuild restarted OneCycle's warmup mid-training,
spiking the learning rate on a converged model. Retrain with resume off:

```python
%cd "/content/zintoo/zintoo ai"
!rm -f "/content/drive/MyDrive/zintoo/models_artifacts/last.pt" \
       "/content/drive/MyDrive/zintoo/models_artifacts/best.pt"
import os
os.environ["ZINTOO_RESUME"] = "false"
os.environ["ZINTOO_EPOCHS"] = "12"
!python -m ml.train
```

**Ship gate.** Do not deploy unless `ml.evaluate` reports:

- `masterCategory` top-1 **> 0.90**
- `gender` top-1 **> 0.85**
- `articleType` macro-F1 **> 0.55**

---

## Why not Streamlit (as the host)

Streamlit re-runs your whole script top-to-bottom on every widget interaction.
Zintoo is a FastAPI server with an SSE event stream, a WebSocket endpoint,
bearer auth, a SQLite database, an autonomous agent loop, and a hand-built
dashboard. Streamlit cannot serve any of that.

Hosting Zintoo *on* Streamlit means deleting the dashboard, the real-time feed,
the agent UI and the auth, then rebuilding a worse version. You'd be throwing
away most of the project.

**So:** host the real app on Hugging Face Spaces (below), and — if you still
want a Streamlit link — point a thin Streamlit client at it. Best of both.

| | Render free | **HF Spaces free** | Streamlit Cloud |
|---|---|---|---|
| RAM | 512 MB (vision OOMs) | **16 GB** | ~1 GB |
| Runs FastAPI + SSE + WS | yes | **yes** | no |
| Docker | paid | **free** | no |
| Sleeps when idle | yes | after 48 h | yes |
| Cost | free | **free** | free |

HF Spaces' 16 GB makes vision comfortable. That's the deployment target.

---

# Part A — Deploy the real app to Hugging Face Spaces

## A1. Create the Space

1. [huggingface.co](https://huggingface.co) → sign up (free)
2. Top-right avatar → **New Space**
3. Fill in:
   - **Space name:** `zintoo-ai`
   - **License:** `mit`
   - **SDK:** **Docker** → **Blank**
   - **Hardware:** `CPU basic · 2 vCPU · 16 GB` (free)
   - **Visibility:** Public
4. **Create Space**

## A2. Prepare your local repo

```cmd
cd "C:\Users\aarus\Downloads\zintoo-ai-vision\zintoo ai"
copy README_HF.md README.md /Y
```

HF reads the YAML front-matter at the top of `README.md` (`sdk: docker`,
`app_port: 7860`). Without it the Space won't build.

Your `models_artifacts\` must contain the four artifacts, and you need
`styles.csv` in the repo so the container can import the catalogue on cold start:

```cmd
mkdir data\fashion-data
copy C:\Users\aarus\Downloads\fashion-data\styles.csv data\fashion-data\
```

Then tell the container where to find it — add to your `Dockerfile` env block
(already present as a default you can override in Space settings):

```
ZINTOO_DATASET_DIR=/app/data/fashion-data
```

## A3. Large files need Git LFS

`catalog_embeddings.npy` is ~22 MB. Plain git will choke on it.

```cmd
git lfs install
git lfs track "*.npy"
git lfs track "*.onnx"
git add .gitattributes
```

## A4. Push

```cmd
git init
git add .
git commit -m "Zintoo AI v2 with vision"
git remote add hf https://huggingface.co/spaces/<your-username>/zintoo-ai
git push hf main --force
```

It'll ask for credentials. Username = your HF username. Password = an **access
token** (Settings → Access Tokens → New token, role `write`). Not your password.

## A5. Set the secret

In your Space: **Settings** → **Variables and secrets** → **New secret**

- Name: `ZINTOO_SECRET_KEY`
- Value: any long random string

Without it, login tokens are invalidated on every restart.

## A6. Watch the build

The **Logs** tab shows the Docker build (~3 min), then:

```
[bootstrap] imported 44,424 products into /app/runtime/zintoo.db
[bootstrap] visual index ids resolve (100%)
zintoo.vision: vision model loaded in 82 ms
zintoo.main: startup complete — ready to serve
```

If bootstrap warns `only 0% of visual-index ids resolve`, your
`catalog_embeddings.npy` was built against a different catalogue — rebuild it
with `ml.build_index`.

Your app is live at `https://<username>-zintoo-ai.hf.space`.

## A7. ⚠️ Before sharing the link

The demo credentials `admin@zintoo.ai / admin123` are **public in your repo**.
Anyone can log in and trigger the inventory agent.

Edit `app/seed.py` → `_seed_users()` and change the passwords, or read them from
env vars, before you give the URL to anyone.

---

# Part B — Optional: a Streamlit link for the demo

This is a **thin client**. It loads no model — it calls your Space over HTTP.

## B1. Push to GitHub

`streamlit_app.py` and `requirements-streamlit.txt` are already in the repo.

```cmd
git remote add origin https://github.com/aarushnalwaya/zintoo-ai.git
git push origin main
```

## B2. Deploy

1. [share.streamlit.io](https://share.streamlit.io) → **New app**
2. Repository: `aarushnalwaya/zintoo-ai`, branch `main`
3. **Main file path:** `streamlit_app.py`
4. **Advanced settings → Secrets**, paste:

```toml
ZINTOO_API_URL = "https://<your-username>-zintoo-ai.hf.space"
```

5. **Deploy**

Streamlit Cloud installs from `requirements.txt` by default — which pulls
FastAPI and onnxruntime you don't need. Either rename
`requirements-streamlit.txt` → `requirements.txt` in a client-only branch, or
accept the slower build.

## B3. Test locally first

```cmd
set ZINTOO_API_URL=http://localhost:8000
python -m pip install -r requirements-streamlit.txt
streamlit run streamlit_app.py
```

---

## Architecture you end up with

```
   Streamlit Cloud                 Hugging Face Spaces
   ┌───────────────┐   HTTPS       ┌──────────────────────────┐
   │ streamlit_app │ ────────────► │ FastAPI  (the real app)  │
   │  upload demo  │  /recommend   │  • ONNX Runtime          │
   └───────────────┘    /image     │  • SQLite (44k catalogue)│
                                   │  • SSE + WebSocket       │
   Anyone with the URL ──────────► │  • dashboard, agent      │
                                   └──────────────────────────┘
```

One model. One database. Two front doors.

---

## Troubleshooting

| Symptom | Fix |
|---|---|
| Space build fails: "no app_port" | `README.md` missing YAML front-matter. Copy `README_HF.md` |
| `Permission denied: /app/runtime` | Don't edit the Dockerfile's `USER appuser` / `chown` lines |
| Space starts, vision `503` | `models_artifacts/` not pushed. Check Git LFS tracked `.onnx`/`.npy` |
| Predictions work, grid empty | Index ids don't resolve. Re-run `ml.build_index`, re-push |
| Images 404 in the grid | Expected — set `ZINTOO_IMAGES_DIR`, or accept placeholders |
| Streamlit: "Cannot reach backend" | Wrong `ZINTOO_API_URL` secret, or the Space is asleep. Open the Space URL once to wake it |
| Space sleeps | Free Spaces idle out after 48 h. Open the URL to wake |

---

## Before real clients use this

1. **Change the demo passwords.** They're in the repo.
2. **The model learned from clean studio shots on white backgrounds.** Real
   customer phone photos — bad lighting, clutter, worn garments — will do
   noticeably worse. Fine-tune on in-domain images before promising accuracy.
3. **Show the confidence score and threshold it.** The Streamlit client already
   flags predictions below 40% as unreliable. Do the same in any client UI.
4. **Long-tail classes were dropped** (<50 samples). The model cannot predict
   them and will confidently pick a neighbour instead.
5. **Report accuracy from `ml/evaluate.py`**, and read its "ten weakest classes"
   list before you show anyone a demo.
