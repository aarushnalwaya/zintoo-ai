import os, asyncio, tempfile, time
os.environ["ZINTOO_DATA_DIR"] = tempfile.mkdtemp()
os.environ["ZINTOO_WEATHER_ENABLED"] = "false"   # network is off; fail-soft path
os.environ["ZINTOO_SECRET_KEY"] = "test-secret-please-change"
os.environ["ZINTOO_ENV"] = "development"

from app import seed, recommender, forecast, orchestrator, auth, db
from app.events import bus, Event

print("=== 1. SEED ===")
t0=time.time(); seed.seed_if_empty(); print(f"seed took {time.time()-t0:.2f}s")
print("products:", db.table_count("products"),
      "| inventory:", db.table_count("inventory"),
      "| demand:", db.table_count("demand_history"),
      "| users:", db.table_count("users"))
assert db.table_count("products") > 0
assert db.table_count("demand_history") > 0

print("\n=== 2. AUTH ===")
h, s = auth.hash_password("admin123")
assert auth.verify_password("admin123", h, s)
assert not auth.verify_password("wrong", h, s)
tok = auth.issue_token("admin@zintoo.ai", "Owner")
payload = auth.decode_token(tok)
assert payload and payload["sub"] == "admin@zintoo.ai" and payload["role"] == "Owner"
assert auth.decode_token(tok + "x") is None, "tampered token must fail"
assert auth.decode_token("garbage.sig") is None
print("password hash/verify OK; token sign/verify/tamper OK")
# verify DB user login path
row = db.query_one("SELECT * FROM users WHERE email=?", ("admin@zintoo.ai",))
assert auth.verify_password("admin123", row["password_hash"], row["salt"])
print("DB-seeded admin login verifies OK")

print("\n=== 3. RECOMMENDER (real TF-IDF) ===")
t0=time.time(); recommender.index.build(); print(f"index build {time.time()-t0:.3f}s over {len(recommender.index.docs)} docs")
res = recommender.index.search("black casual shoes for men", top_k=5)
print("query 'black casual shoes for men' ->", len(res), "results")
for r in res[:3]:
    print(f"   #{r['rank']} score={r['similarity_score']} :: {r['name']} [{r['gender']}/{r['usage']}]")
assert len(res) > 0
# singular queries must hit plural catalog terms (stemming)
for q in ["jacket", "shoe", "watch", "jean"]:
    assert len(recommender.index.search(q, 3)) > 0, f"singular query {q!r} returned nothing"
print("singular/plural stemming OK")
# colour must dominate ranking, not be diluted across fields
top = recommender.index.search("black casual shoes for men", top_k=1)[0]
assert top["color"] == "Black" and top["article_type"] == "Casual Shoes", f"bad top hit: {top['name']}"
print(f"ranking OK -> top hit {top['name']!r}")
# gender filter: NON-VACUOUS (must return rows), excludes wrong gender, allows Unisex
res_w = recommender.index.search("jacket", top_k=20, filters={"gender":"Women"})
assert len(res_w) > 0, "gender filter returned nothing (vacuous-pass bug)"
assert not any(r["gender"] == "Men" for r in res_w), "gender filter leaked men's items"
assert any(r["gender"] == "Women" for r in res_w)
print(f"gender filter -> {len(res_w)} results, correctly scoped")
# nonsense must return nothing, not junk
assert recommender.index.search("zzzz qqqq", 5) == [], "nonsense query returned junk"
print("nonsense query correctly returns 0 results")

print("\n=== 4. FORECAST (real, from history) ===")
skus = forecast.available_skus(); pins = forecast.available_pincodes()
print("forecastable skus:", len(skus), "pincodes:", pins)
fc = forecast.forecast(skus[0], pins[0], hours=24)
print(f"SKU {skus[0]} @ {pins[0]}: total={fc['predicted_total_demand']} "
      f"peak={fc['peak_hour_demand']} @ {fc['peak_hour'][11:16]} "
      f"| MAPE={fc['metrics']['mape']}% RMSE={fc['metrics']['rmse']} | src={fc['source']}")
assert len(fc["hourly_forecast"]) == 24
assert all(h["lower_bound"] <= h["predicted_demand"] <= h["upper_bound"] for h in fc["hourly_forecast"])
try:
    forecast.forecast("NOPE_SKU", "999999", 24); print("ERROR: should have raised")
except ValueError as e: print("unknown SKU correctly raises ValueError")

print("\n=== 5. ORCHESTRATOR (stateful, atomic, events) ===")
async def run():
    events=[]
    q = await bus.subscribe()
    async def drain():
        try:
            while True:
                e = await asyncio.wait_for(q.get(), timeout=0.5); events.append(e)
        except asyncio.TimeoutError: pass
    before = orchestrator.inventory_summary()
    tot_before = sum(w["total_stock"] for w in before)
    report = await orchestrator.run_cycle()
    await drain()
    after = orchestrator.inventory_summary()
    tot_after = sum(w["total_stock"] for w in after)
    print("status:", report["status"], "| transfers:", len(report["transfers"]))
    ok = [t for t in report["transfers"] if t.get("success")]
    print(f"successful transfers: {len(ok)} | events emitted: {len(events)}")
    print("event types:", sorted(set(e.type for e in events)))
    # conservation of stock: transfers move stock, never create/destroy it
    assert tot_before == tot_after, f"stock not conserved: {tot_before} -> {tot_after}"
    print(f"stock conserved across cycle: {tot_before} == {tot_after}")
    # persistence: transfers table populated + run recorded
    print("persisted transfers:", db.table_count("transfers"),
          "| runs:", db.table_count("orchestration_runs"))
    assert db.table_count("orchestration_runs") >= 1
    stats = orchestrator.run_stats(); print("run_stats:", stats)
    # idempotent-ish second run should still conserve
    r2 = await orchestrator.run_cycle()
    a2 = sum(w["total_stock"] for w in orchestrator.inventory_summary())
    assert a2 == tot_after, "second cycle broke conservation"
    print("second cycle status:", r2["status"], "| stock still", a2)
asyncio.run(run())

print("\n=== 6. EVENT BUS SSE FORMAT ===")
e = Event(type="agent.act", data={"message":"hi"}, id=7)
sse = e.to_sse()
assert sse.startswith("id: 7") and "event: agent.act" in sse and sse.endswith("\n\n")
print("SSE frame well-formed:\n" + "".join("   "+l+"\n" for l in sse.strip().split("\n")))

print("\nALL CORE TESTS PASSED ✅")
