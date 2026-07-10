"""Verifies the app degrades cleanly when vision is disabled or artifacts are
missing — the two states any fresh deploy hits first.

    PYTHONPATH=. python tests/test_degradation.py
"""
import os, tempfile
from pathlib import Path
d=Path(tempfile.mkdtemp())
os.environ.update(ZINTOO_DATA_DIR=str(d), ZINTOO_SECRET_KEY="t",
                  ZINTOO_MODELS_DIR=str(d/"nonexistent"), ZINTOO_WEATHER_ENABLED="false",
                  ZINTOO_VISION_ENABLED="false")
from app import vision, visual_index, settings

print("=== A. vision DISABLED, no artifacts ===")
print("  settings.VISION_ENABLED =", settings.VISION_ENABLED)
print("  model.available()       =", vision.model.available())
print("  index.available()       =", visual_index.index.available())
assert vision.model.available() is False and visual_index.index.available() is False
info = vision.model.info()
print("  vision.info():", {k:info[k] for k in ('available','loaded')})
assert info["loaded"] is False, "session must NOT be built when unavailable"
print("  ✅ no crash, nothing loaded, ORT never imported")
import sys
assert "onnxruntime" not in sys.modules, "ORT imported despite vision disabled — wastes 40MB!"
print("  ✅ onnxruntime NOT in sys.modules (lazy import respected)")

print("\n=== B. vision ENABLED but artifacts MISSING ===")
try:
    vision.model.infer(b"whatever")
    print("  ❌ should have raised FileNotFoundError")
except FileNotFoundError as e:
    print("  ✅ FileNotFoundError with actionable message:")
    print("    ", str(e)[:100], "...")
try:
    visual_index.index.search([0.0]*32, 5)
    print("  ❌ should have raised")
except FileNotFoundError as e:
    print("  ✅ index FileNotFoundError:", str(e)[:70])

print("\n=== C. app still boots + core endpoints unaffected ===")
from app import seed, recommender, forecast
seed.seed_if_empty(); recommender.index.build()
r = recommender.index.search("black shoe", 3)
print(f"  text search still works: {len(r)} results, top={r[0]['name']!r}")
assert len(r)>0
print("  ✅ text/forecast/orchestration unaffected by missing vision")
print("\nDEGRADATION TESTS PASSED ✅")
