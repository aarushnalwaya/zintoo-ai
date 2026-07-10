"""
Zintoo Vision — Streamlit client.

This is a THIN CLIENT. It does not host the model, the database, or the agent.
It calls the deployed FastAPI backend over HTTP.

Why it's built this way:
  Streamlit re-runs the whole script top-to-bottom on every widget interaction.
  Loading a 4 MB ONNX model, a 22 MB embedding index and a 44k-row SQLite
  catalogue into that lifecycle is wasteful and fragile, and Streamlit cannot
  serve the app's real-time SSE stream, WebSocket endpoint, bearer auth, or the
  existing dashboard at all. So the backend stays where it belongs, and this
  gives you a shareable Streamlit URL for the image-classification demo.

Deploy on Streamlit Community Cloud:
  1. Push this repo to GitHub.
  2. share.streamlit.io -> New app -> main file: streamlit_app.py
  3. Advanced settings -> Secrets, paste:
         ZINTOO_API_URL = "https://<your-space>.hf.space"

Run locally:
    pip install -r requirements-streamlit.txt
    ZINTOO_API_URL=http://localhost:8000 streamlit run streamlit_app.py
"""

from __future__ import annotations

import os

import requests
import streamlit as st

st.set_page_config(page_title="Zintoo Vision", page_icon="👗", layout="wide")

DEFAULT_API = "http://localhost:8000"


def api_url() -> str:
    # st.secrets raises if no secrets.toml exists locally — don't let that crash.
    try:
        if "ZINTOO_API_URL" in st.secrets:
            return str(st.secrets["ZINTOO_API_URL"]).rstrip("/")
    except Exception:
        pass
    return os.getenv("ZINTOO_API_URL", DEFAULT_API).rstrip("/")


API = api_url()

st.title("👗 Zintoo Vision")
st.caption("Fashion image classification + visual similarity search")

with st.sidebar:
    st.subheader("Backend")
    st.code(API, language=None)
    if st.button("Check status"):
        try:
            r = requests.get(f"{API}/vision/health", timeout=10)
            r.raise_for_status()
            h = r.json()
            if h.get("enabled") and h.get("model", {}).get("available"):
                st.success("Vision online")
            else:
                st.warning("Backend reachable, vision disabled or model missing")
            st.json(h)
        except requests.RequestException as exc:
            st.error(f"Cannot reach backend: {exc}")
    st.markdown("---")
    top_k = st.slider("Similar products", 3, 20, 8)
    st.caption(
        "The full dashboard — real-time agent feed, inventory orchestration, "
        "demand forecasting — lives on the backend, not here."
    )

uploaded = st.file_uploader(
    "Upload a garment photo", type=["jpg", "jpeg", "png", "webp"],
    help="Max 8 MB. Studio-style product shots work best.",
)

if uploaded is None:
    st.info("Upload an image to classify it and find visually similar products.")
    st.stop()

if uploaded.size > 8 * 1024 * 1024:
    st.error(f"Image is {uploaded.size / 1e6:.1f} MB — the limit is 8 MB.")
    st.stop()

left, right = st.columns([1, 2])
with left:
    st.image(uploaded, caption=uploaded.name, use_container_width=True)

with st.spinner("Classifying…"):
    try:
        resp = requests.post(
            f"{API}/recommend/image",
            params={"top_k": top_k},
            files={"file": (uploaded.name, uploaded.getvalue(), uploaded.type)},
            timeout=60,
        )
    except requests.RequestException as exc:
        st.error(f"Could not reach the backend at {API}\n\n{exc}")
        st.stop()

if resp.status_code == 503:
    st.warning(resp.json().get("detail", "Vision is disabled on the backend."))
    st.stop()
if resp.status_code == 400:
    st.error(resp.json().get("detail", "That image could not be read."))
    st.stop()
if not resp.ok:
    st.error(f"Backend returned HTTP {resp.status_code}")
    st.stop()

data = resp.json()

with right:
    st.subheader("Predicted attributes")
    preds = data.get("predictions", {})
    cols = st.columns(max(1, len(preds)))
    for col, (task, items) in zip(cols, preds.items()):
        with col:
            st.markdown(f"**{task}**")
            for i, p in enumerate(items):
                conf = p["confidence"]
                st.progress(min(1.0, conf), text=f"{p['label']} — {conf:.0%}")
                if i == 0 and conf < 0.40:
                    st.caption("⚠️ Low confidence — treat as unreliable.")

    t = data.get("timing_ms", {})
    st.caption(
        f"inference {t.get('inference', 0):.1f} ms · "
        f"preprocess {t.get('preprocess', 0):.1f} ms · "
        f"total {t.get('total', 0):.1f} ms"
    )

st.markdown("---")
results = data.get("results", [])
st.subheader(f"Visually similar products ({len(results)})")

if not results:
    st.warning(
        "No similar products returned. If the backend has a visual index, this "
        "usually means the index's product ids don't resolve against its "
        "catalogue — run `python -m ml.import_catalog` there."
    )
else:
    for row in range(0, len(results), 4):
        for col, item in zip(st.columns(4), results[row:row + 4]):
            with col:
                img = f"{API}/images/{item['product_id']}.jpg"
                st.image(img, use_container_width=True)
                st.markdown(f"**{item['name'][:40]}**")
                st.caption(
                    f"{item['article_type']} · {item['color']} · {item['gender']}\n\n"
                    f"similarity {item['similarity_score']:.3f}"
                )

st.markdown("---")
st.caption(
    "Trained on the Kaggle Fashion Product Images dataset — clean studio shots on "
    "white backgrounds. Accuracy drops on real-world photos with cluttered "
    "backgrounds or worn garments. Check the confidence score before trusting a label."
)
