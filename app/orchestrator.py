"""
Inventory orchestration agent (ReAct-style, rule-based) — now stateful.

The original mutated an in-memory DataFrame per request and discarded it, so
transfers never persisted and every "cycle" started from the same CSV. This
version reads and writes the SQLite inventory transactionally, records each
transfer and run, and publishes real-time events (OBSERVE / THINK / ACT /
REFLECT) so the live dashboard reflects true server state.
"""

from __future__ import annotations

from datetime import datetime, timezone

from . import db, settings
from .events import bus
from .observability import get_logger, metrics

log = get_logger("zintoo.orchestrator")


def _now() -> str:
    return datetime.now(timezone.utc).isoformat()


def inventory_summary() -> list[dict]:
    rows = db.query(
        """
        SELECT warehouse_id,
               COUNT(DISTINCT sku)                      AS total_skus,
               SUM(current_stock)                       AS total_stock,
               ROUND(AVG(current_stock), 1)             AS avg_stock,
               MIN(current_stock)                       AS min_stock,
               SUM(CASE WHEN current_stock <= reorder_threshold THEN 1 ELSE 0 END) AS low_stock_skus
        FROM inventory
        GROUP BY warehouse_id
        ORDER BY warehouse_id
        """
    )
    return [dict(r) for r in rows]


def stock_matrix() -> list[dict]:
    rows = db.query(
        "SELECT warehouse_id, sku, pincode, current_stock, reorder_threshold, max_capacity "
        "FROM inventory ORDER BY sku, warehouse_id"
    )
    return [dict(r) for r in rows]


def recent_transfers(limit: int = 25) -> list[dict]:
    rows = db.query(
        "SELECT ts, sku, from_warehouse, to_warehouse, quantity, priority, reason, success "
        "FROM transfers ORDER BY id DESC LIMIT ?",
        (limit,),
    )
    return [dict(r) for r in rows]


def _status(stock: int, threshold: int) -> str:
    if stock <= threshold // 2:
        return "critical"
    if stock <= threshold:
        return "low"
    return "healthy"


async def run_cycle() -> dict:
    """Execute one OBSERVE -> THINK -> ACT -> REFLECT rebalancing cycle."""
    metrics.inc("orchestration_runs_total")
    decision_log: list[dict] = []
    step = 0

    def logd(phase: str, action: str, details: dict, reasoning: str) -> None:
        nonlocal step
        step += 1
        entry = {
            "step": step,
            "timestamp": _now(),
            "phase": phase,
            "action": action,
            "details": details,
            "reasoning": reasoning,
        }
        decision_log.append(entry)

    # ── OBSERVE ──────────────────────────────────────────────────────
    rows = stock_matrix()
    by_sku: dict[str, list[dict]] = {}
    for r in rows:
        r = dict(r)
        r["status"] = _status(r["current_stock"], r["reorder_threshold"])
        r["surplus"] = max(0, r["current_stock"] - r["reorder_threshold"] * 2)
        r["deficit"] = max(0, r["reorder_threshold"] - r["current_stock"])
        by_sku.setdefault(r["sku"], []).append(r)

    needs = [r for group in by_sku.values() for r in group if r["deficit"] > 0]
    surpluses = [r for group in by_sku.values() for r in group if r["surplus"] > 0]
    critical = sum(1 for r in needs if r["status"] == "critical")
    logd("OBSERVE", "inventory_scan",
         {"needs": len(needs), "surplus_locations": len(surpluses), "critical": critical},
         "Scanned all warehouse-SKU pairs for stockout risk.")
    await bus.publish("agent.observe", {
        "message": f"Scanned {len(rows)} SKU-warehouse pairs — {len(needs)} need restock ({critical} critical)",
        "needs": len(needs), "critical": critical,
    })

    # ── THINK ────────────────────────────────────────────────────────
    surplus_by_sku: dict[str, list[dict]] = {}
    for s in surpluses:
        surplus_by_sku.setdefault(s["sku"], []).append(s)

    plan = []
    for need in sorted(needs, key=lambda n: n["deficit"], reverse=True):
        sources = surplus_by_sku.get(need["sku"])
        if not sources:
            logd("THINK", "no_source", {"sku": need["sku"], "wh": need["warehouse_id"]},
                 f"No surplus warehouse for {need['sku']}; external procurement needed.")
            continue
        source = max(sources, key=lambda s: s["surplus"])
        qty = min(need["deficit"] + settings.REORDER_THRESHOLD, source["surplus"], 30)
        headroom = need["max_capacity"] - need["current_stock"]
        qty = min(qty, headroom)
        if qty <= 0:
            continue
        plan.append({
            "sku": need["sku"],
            "from_warehouse": source["warehouse_id"],
            "to_warehouse": need["warehouse_id"],
            "quantity": qty,
            "priority": "high" if need["status"] == "critical" else "medium",
            "reason": (
                f"{need['warehouse_id']} at {need['current_stock']} units "
                f"(threshold {need['reorder_threshold']}); source {source['warehouse_id']} "
                f"has surplus {source['surplus']}."
            ),
        })
        source["surplus"] -= qty
    logd("THINK", "transfer_plan",
         {"planned": len(plan), "units": sum(p["quantity"] for p in plan)},
         f"Matched {len(plan)} surplus->deficit transfers.")
    await bus.publish("agent.think", {
        "message": f"Planned {len(plan)} transfers ({sum(p['quantity'] for p in plan)} units)",
        "planned": len(plan),
    })

    if not plan:
        db.execute(
            "INSERT INTO orchestration_runs(ts, status, transfers, units_moved, cost) "
            "VALUES(?,?,?,?,?)", (_now(), "balanced", 0, 0, 0.0),
        )
        await bus.publish("agent.reflect", {"message": "Inventory balanced — no transfers needed", "status": "balanced"})
        return {"status": "balanced", "transfers": [], "decision_log": decision_log, "reflection": None}

    # ── ACT (atomic per transfer) ────────────────────────────────────
    executed = []
    for p in plan:
        try:
            with db.transaction() as conn:
                src = conn.execute(
                    "SELECT current_stock FROM inventory WHERE warehouse_id=? AND sku=?",
                    (p["from_warehouse"], p["sku"]),
                ).fetchone()
                dst = conn.execute(
                    "SELECT current_stock, max_capacity FROM inventory WHERE warehouse_id=? AND sku=?",
                    (p["to_warehouse"], p["sku"]),
                ).fetchone()
                if not src or not dst or src["current_stock"] < p["quantity"]:
                    raise ValueError("insufficient source stock")
                if dst["current_stock"] + p["quantity"] > dst["max_capacity"]:
                    raise ValueError("would exceed destination capacity")
                conn.execute(
                    "UPDATE inventory SET current_stock = current_stock - ? WHERE warehouse_id=? AND sku=?",
                    (p["quantity"], p["from_warehouse"], p["sku"]),
                )
                conn.execute(
                    "UPDATE inventory SET current_stock = current_stock + ? WHERE warehouse_id=? AND sku=?",
                    (p["quantity"], p["to_warehouse"], p["sku"]),
                )
                conn.execute(
                    "INSERT INTO transfers(ts, sku, from_warehouse, to_warehouse, quantity, priority, reason, success) "
                    "VALUES(?,?,?,?,?,?,?,1)",
                    (_now(), p["sku"], p["from_warehouse"], p["to_warehouse"], p["quantity"], p["priority"], p["reason"]),
                )
            p["success"] = True
            metrics.inc("transfers_executed_total")
        except Exception as exc:  # noqa: BLE001
            p["success"] = False
            p["error"] = str(exc)
            db.execute(
                "INSERT INTO transfers(ts, sku, from_warehouse, to_warehouse, quantity, priority, reason, success) "
                "VALUES(?,?,?,?,?,?,?,0)",
                (_now(), p["sku"], p["from_warehouse"], p["to_warehouse"], p["quantity"], p["priority"], str(exc)),
            )
        executed.append(p)
        logd("ACT", "execute_transfer", {k: p[k] for k in ("sku", "from_warehouse", "to_warehouse", "quantity", "success")}, p["reason"])
        await bus.publish("agent.act", {
            "message": (
                f"{'✅' if p['success'] else '❌'} {p['quantity']}× {p['sku']}: "
                f"{p['from_warehouse']} → {p['to_warehouse']} ({p['priority']})"
            ),
            "transfer": {k: p[k] for k in ("sku", "from_warehouse", "to_warehouse", "quantity", "priority", "success")},
        })

    # ── REFLECT ──────────────────────────────────────────────────────
    ok = [p for p in executed if p["success"]]
    units = sum(p["quantity"] for p in ok)
    cost = units * settings.TRANSFER_COST_PER_UNIT
    db.execute(
        "INSERT INTO orchestration_runs(ts, status, transfers, units_moved, cost) VALUES(?,?,?,?,?)",
        (_now(), "completed", len(ok), units, cost),
    )
    reflection = {
        "total_transfers_attempted": len(executed),
        "successful_transfers": len(ok),
        "failed_transfers": len(executed) - len(ok),
        "total_units_moved": units,
        "estimated_cost": cost,
        "post_transfer_summary": inventory_summary(),
    }
    logd("REFLECT", "post_transfer_assessment", reflection,
         f"Executed {len(ok)}/{len(executed)} transfers, moved {units} units at ₹{cost:.0f}.")
    await bus.publish("agent.reflect", {
        "message": f"Cycle complete — {len(ok)} transfers, {units} units moved, ₹{cost:.0f}",
        "reflection": {k: reflection[k] for k in ("successful_transfers", "failed_transfers", "total_units_moved", "estimated_cost")},
    })

    return {"status": "completed", "transfers": executed, "decision_log": decision_log, "reflection": reflection}


def run_stats() -> dict:
    row = db.query_one(
        "SELECT COUNT(*) AS cycles, COALESCE(SUM(transfers),0) AS transfers, "
        "COALESCE(SUM(units_moved),0) AS units FROM orchestration_runs"
    )
    return {"cycles": row["cycles"], "transfers": row["transfers"], "units_moved": row["units"]}
