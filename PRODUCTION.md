# Zintoo AI — Production Upgrade (v2)

This document explains what was wrong with the deployed app, what changed, and
how to run and deploy the production build.

---

## 1. TL;DR — what was actually broken

The live site *looked* like it worked, but it was a facade:

| Symptom | Root cause |
|---|---|
| Every intelligence endpoint returned **HTTP 500** in production | `render.yaml` built from `requirements-deploy.txt`, which omits `torch`, `faiss`, `prophet`, `pandas`, `numpy`. The first `import` inside `/recommend`, `/forecast`, `/orchestrate` threw `ImportError`. |
| …but the UI still "worked" | The frontend wrapped every `fetch` in a `catch` that silently swapped in **hard-coded mock data** (random Unsplash photos, a client-side sine-wave forecast, static tables). |
| "Real-time" Agent Feed | A `setInterval` cycling **7 hard-coded strings every 15 s**. No WebSocket/SSE anywhere. |
| No persistence | The inventory toolkit read a CSV into a DataFrame, mutated it in memory, and discarded it per request. Transfers never persisted. |
| No data at all | `warehouse_inventory.csv`, `demand_history.csv`, the FAISS index and product map were never shipped in the repo. |
| Auth did nothing | Login minted a token that was **never validated** on any later request; no endpoint was protected. Passwords were unsalted SHA-256. |
| Error handling leaked internals | Every handler did `raise HTTPException(500, str(e))` — raw exception text to clients, no logging, no request correlation. |
| Invalid CORS | `allow_origins=["*"]` **with** `allow_credentials=True` — browsers reject this combination. |
| Won't scale on Render | The *real* ML path (FashionCLIP ≈ 600 MB + torch) can't fit the free tier's 512 MB RAM. That's why it was stripped — which is what made it a facade. |

> **Vision:** image classification and visual search are documented separately in **[VISION.md](VISION.md)**.

## 2. Design decision: fit the free tier honestly

Rather than fake results, v2 delivers **genuinely working** features that fit in
512 MB with fast cold starts, by replacing the heavyweight stack with
dependency-light equivalents:

| Capability | Before | After |
|---|---|---|
| Text recommendation | FashionCLIP + FAISS (won't fit) → 500 | Pure-Python **TF-IDF** cosine retrieval over a seeded catalog (~8 ms build, few MB RAM) |
| Image / multimodal | 500 | **Real** ONNX classifier + visual similarity search — see `VISION.md` |
| Forecasting | Prophet + missing CSV → 500 | Pure-Python **seasonal profile + Holt trend + weather regressor** over real seeded history, with MAPE/RMSE backtest |
| Inventory/orchestration | in-memory DataFrame, no persistence | **SQLite**-backed, transactional, atomic transfers, persisted history |
| Real-time | fake `setInterval` | **SSE** (`/events`) + **WebSocket** (`/ws`) driven by an in-process event bus |
| Data | none shipped | **Self-seeding** SQLite (deterministic) on first boot |

Runtime dependencies are now just `fastapi`, `uvicorn[standard]`, `python-multipart`,
`pydantic`. Everything else is Python stdlib (`sqlite3`, `hashlib`, `hmac`, `urllib`).

## 3. New architecture

```
                         Browser (SPA dashboard)
        HTTP (REST + JSON)      │      SSE  /events   +   WS  /ws
                                ▼
                       ┌──────────────────┐
                       │  FastAPI (app/)  │  1 uvicorn worker
                       └──────────────────┘
   middleware.py  ── request-id · timing · metrics · security headers
   security.py    ── Bearer auth · role guard · per-IP rate limit
   events.py      ── async event bus → SSE + WebSocket fan-out
        │
        ├── recommender.py   TF-IDF index (in-memory, built at startup)
        ├── forecast.py      seasonal + Holt + weather regressor
        ├── orchestrator.py  ReAct rebalancing, atomic, event-emitting
        ├── weather.py       Open-Meteo: timeout + retry + TTL cache, fail-soft
        └── db.py ─────────► SQLite (WAL) ◄── seed.py (deterministic first-boot)
```

Module map (all new code under `app/`):

- `settings.py` — env-driven config, safe defaults, `public_config()` for `/health`.
- `observability.py` — JSON/pretty logging, request-id contextvar, Prometheus-text metrics. **No framework coupling.**
- `middleware.py` — request context middleware + global exception handler.
- `db.py` — SQLite connection mgmt (WAL, busy_timeout), schema, parameterized queries, transactions.
- `seed.py` — idempotent seeding: users, 400-item catalog, inventory, 90-day hourly demand.
- `auth.py` — PBKDF2 hashing, HMAC-signed tokens, token-bucket limiter (pure).
- `security.py` — FastAPI auth/role/rate-limit dependencies.
- `events.py` — event bus, `Event`, SSE stream generator.
- `weather.py`, `forecast.py`, `recommender.py`, `orchestrator.py` — domain logic.
- `main.py` — app wiring, lifespan, routes.

The original research code (`api/`, `models/`, `agents/`, `data/`, `notebooks/`)
is retained unchanged for reference and for the optional ML tier; **it is no
longer on the serving path.** Deployment runs `app.main:app`.

## 4. API surface

| Endpoint | Method | Auth | Notes |
|---|---|---|---|
| `/` | GET | – | Dashboard SPA |
| `/health` | GET | – | Liveness + real subsystem status + config |
| `/readiness` | GET | – | DB reachability (503 if not) |
| `/metrics` | GET | – | Prometheus text exposition |
| `/api/login` | POST | – | Returns a signed Bearer token |
| `/api/me` | GET | Bearer | Current token identity |
| `/recommend` | POST | rate-limited | Real TF-IDF text search |
| `/recommend/image` | POST | – | `501 requires_vision_tier` (honest) |
| `/catalog` | GET | – | Paginated product catalog |
| `/forecast/{sku}/{pincode}` | GET | – | Real forecast + intervals + metrics |
| `/skus` | GET | – | Forecastable SKUs + pincodes |
| `/inventory/summary` | GET | – | Per-warehouse rollup |
| `/inventory/stock` | GET | – | Full stock matrix |
| `/inventory/transfers` | GET | – | Recent persisted transfers |
| `/orchestrate` | POST | **Bearer + role** | Runs a rebalancing cycle; emits events |
| `/orchestrate/stats` | GET | – | Cumulative run stats |
| `/events` | GET (SSE) | – | Live event stream |
| `/ws` | WS | – | Live event stream (bidirectional) |

Interactive docs at `/api/docs`.

## 5. Environment variables

All optional for local dev (safe defaults). See `.env.example`. In production
set at least `ZINTOO_SECRET_KEY`.

| Var | Default | Purpose |
|---|---|---|
| `ZINTOO_ENV` | `development` | `production` enables JSON logs + secret warning |
| `ZINTOO_SECRET_KEY` | random per-boot | HMAC key for tokens. **Set in prod** or tokens die on restart |
| `ZINTOO_TOKEN_TTL` | `43200` (12h) | Token lifetime (s) |
| `ZINTOO_DATA_DIR` | `./runtime` | Where SQLite + data live (point at a mounted disk for durability) |
| `ZINTOO_DB_PATH` | `<DATA_DIR>/zintoo.db` | Explicit DB path override |
| `ZINTOO_JSON_LOGS` | `true` in prod | Structured JSON logs |
| `ZINTOO_LOG_LEVEL` | `INFO` | Log level |
| `ZINTOO_CORS_ORIGINS` | same-origin | Comma-separated allowed origins |
| `ZINTOO_RATE_LIMIT` / `_RPS` / `_BURST` | `true` / `20` / `40` | Per-IP token bucket |
| `ZINTOO_WEATHER_ENABLED` / `_TIMEOUT` / `_CACHE_TTL` | `true` / `4.0` / `900` | Open-Meteo regressor |

## 6. Run locally

```bash
cd "zintoo ai"
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
uvicorn app.main:app --reload --port 8000
# open http://localhost:8000  — login: admin@zintoo.ai / admin123  (or demo@zintoo.ai / demo123)
```

First boot seeds the DB (a few seconds). Watch the Agent Feed and then click
**Run Cycle** / **Optimize Now** — the feed updates live over SSE from real
server events, and inventory changes persist.

## 7. Deploy to Render

The repo is a Render Blueprint.

1. Commit and push these changes to `main`.
2. Render dashboard → **New +** → **Blueprint** → select the repo. Render reads
   `render.yaml`: free plan, `pip install -r requirements.txt`,
   `uvicorn app.main:app`, health check `/health`, and a generated
   `ZINTOO_SECRET_KEY`.
3. Deploy. If you're updating the existing `zintoo-ai` service instead, just
   push — `autoDeploy: true` redeploys on push. (Confirm the service's Start
   Command is `uvicorn app.main:app ...`, not the old `api.main:app`.)

Free-tier notes:
- **1 worker** is required: SSE/WebSocket fan-out and the SQLite/event-bus state
  are in-process, so multiple workers would not share them. Scale vertically
  (bigger plan) before adding workers, and use an external broker (Redis) +
  Postgres if you later need horizontal scale.
- Free instances **sleep after ~15 min idle**; the next request pays a cold
  start. `/health` is cheap to keep warm with an external pinger.
- The free filesystem is **ephemeral** — the DB re-seeds deterministically on
  every boot, so the app always comes up populated. For durable transfer
  history, use a paid plan and uncomment the `disk:` block in `render.yaml`
  (state then lives at `/var/data`).

## 8. What to verify after deploy

```bash
BASE=https://zintoo-ai-1.onrender.com
curl -s $BASE/health | jq          # status:"healthy", all modules true
curl -s $BASE/skus | jq '.skus|length'
curl -s "$BASE/forecast/BACKPA_001/400001?hours=24" | jq '.predicted_total_demand, .metrics'
curl -s -X POST $BASE/recommend -H 'content-type: application/json' \
     -d '{"text_query":"black casual shoes for men","top_k":5}' | jq '.total_results'
# real-time: this streams live events
curl -N $BASE/events
# auth is enforced now:
curl -s -X POST $BASE/orchestrate            # -> 401
TOKEN=$(curl -s -X POST $BASE/api/login -H 'content-type: application/json' \
        -d '{"email":"admin@zintoo.ai","password":"admin123"}' | jq -r .token)
curl -s -X POST $BASE/orchestrate -H "authorization: Bearer $TOKEN" | jq '.status'
```

## 9. Security notes

- Demo credentials (`admin123` / `demo123`) are seeded for the demo. **Change or
  remove them before real client use** — edit `_seed_users()` in `app/seed.py`
  or manage the `users` table.
- Passwords: PBKDF2-HMAC-SHA256, 200k rounds, per-user salt.
- Tokens: HMAC-SHA256 signed, expiry-checked on every protected call. Set a
  strong `ZINTOO_SECRET_KEY`.
- CORS defaults to same-origin. Rate limiting is on by default.
- Errors return a correlated `request_id` instead of stack traces.
