# Tests

`test_smoke.py` is a dependency-free end-to-end check of the core logic
(seed → auth → recommender → forecast → orchestrator → event bus). It runs
against a throwaway SQLite DB and needs no network.

    PYTHONPATH=.. ZINTOO_WEATHER_ENABLED=false python tests/test_smoke.py

For full HTTP-level tests, install `requirements.txt` plus `httpx`/`pytest` and
hit the running app.

`test_vision.py` exercises the vision serving path end-to-end (preprocessing,
ONNX Runtime inference, multi-head decoding, embedding normalisation, visual
similarity search, hostile-input handling, latency). It synthesises a valid
ONNX model via `_onnx_builder.py`, so it needs **no trained model, no torch and
no network**:

    PYTHONPATH=. ZINTOO_WEATHER_ENABLED=false python tests/test_vision.py

Note: the latency it prints uses the stub model. Real MobileNetV3-Small
inference is ~15-30 ms/image on one CPU core; preprocessing (~10 ms) is real.
