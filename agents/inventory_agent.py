"""
╔══════════════════════════════════════════════════════════════╗
║  🤖 AGENTIC ORCHESTRATION — Inventory Agent                 ║
║  Rule-based autonomous inventory rebalancing agent           ║
║  with structured reasoning and decision logging              ║
╚══════════════════════════════════════════════════════════════╝

This agent autonomously:
1. Scans all warehouses for stockout risks
2. Cross-references with demand forecasts
3. Identifies surplus-deficit pairs
4. Executes optimal transfers
5. Logs every decision with reasoning
"""

import sys
import json
from pathlib import Path
from datetime import datetime

import pandas as pd

sys.path.insert(0, str(Path(__file__).parent.parent))
from config import (
    WAREHOUSE_IDS, WAREHOUSE_PINCODE_MAP, REORDER_THRESHOLD,
    SLA_MINUTES, TRANSFER_COST_PER_UNIT, LOGS_DIR,
)
from agents.tools import InventoryToolkit


class InventoryOrchestrationAgent:
    """
    Autonomous inventory orchestration agent.

    Uses a rule-based ReAct-style approach:
    1. OBSERVE → Scan inventory and forecast data
    2. THINK → Identify risks and opportunities
    3. ACT → Execute transfers
    4. REFLECT → Evaluate decisions

    This mimics agentic behavior without requiring an external LLM API key.
    """

    def __init__(self, toolkit: InventoryToolkit = None):
        print("=" * 60)
        print("🤖 AGENTIC ORCHESTRATION: Inventory Agent Initializing")
        print("=" * 60)

        self.toolkit = toolkit or InventoryToolkit()
        self.decision_log = []
        self.step_counter = 0

        print("   ✅ Agent ready for autonomous orchestration")

    def _log_decision(self, phase, action, details, reasoning):
        """Log an agent decision with structured reasoning."""
        self.step_counter += 1
        entry = {
            "step": self.step_counter,
            "timestamp": datetime.now().isoformat(),
            "phase": phase,
            "action": action,
            "details": details,
            "reasoning": reasoning,
        }
        self.decision_log.append(entry)
        return entry

    def _print_step(self, phase, emoji, message):
        """Print a formatted agent step."""
        phase_colors = {
            "OBSERVE": "🔍",
            "THINK": "🧠",
            "ACT": "⚡",
            "REFLECT": "📊",
        }
        icon = phase_colors.get(phase, "➡️")
        print(f"\n   [{self.step_counter + 1}] {icon} {phase}: {message}")

    def observe_inventory_state(self):
        """
        OBSERVE: Scan all warehouses and identify current state.
        Returns a structured assessment.
        """
        self._print_step("OBSERVE", "🔍", "Scanning all warehouse inventory levels...")

        assessment = {
            "critical_skus": [],    # Stock at or below threshold/2
            "low_stock_skus": [],   # Stock below threshold
            "surplus_skus": [],     # Stock well above threshold
            "healthy_skus": [],     # Stock at good levels
        }

        all_skus = self.toolkit.inventory["sku"].unique()

        for sku in all_skus:
            warehouses = self.toolkit.get_all_warehouses_for_sku(sku)

            for wh_info in warehouses:
                record = {"sku": sku, **wh_info}

                if wh_info["status"] == "critical":
                    assessment["critical_skus"].append(record)
                elif wh_info["status"] == "low":
                    assessment["low_stock_skus"].append(record)
                elif wh_info["surplus"] > 0:
                    assessment["surplus_skus"].append(record)
                else:
                    assessment["healthy_skus"].append(record)

        self._log_decision(
            phase="OBSERVE",
            action="inventory_scan",
            details={
                "critical_count": len(assessment["critical_skus"]),
                "low_stock_count": len(assessment["low_stock_skus"]),
                "surplus_count": len(assessment["surplus_skus"]),
                "healthy_count": len(assessment["healthy_skus"]),
                "total_skus": len(all_skus),
            },
            reasoning="Initial scan of all warehouse inventory levels to identify stockout risks.",
        )

        print(f"      Critical: {len(assessment['critical_skus'])} SKU-warehouse pairs")
        print(f"      Low stock: {len(assessment['low_stock_skus'])} SKU-warehouse pairs")
        print(f"      Surplus: {len(assessment['surplus_skus'])} SKU-warehouse pairs")
        print(f"      Healthy: {len(assessment['healthy_skus'])} SKU-warehouse pairs")

        return assessment

    def think_about_transfers(self, assessment):
        """
        THINK: Analyze the inventory state and plan transfers.
        Matches deficit locations with surplus locations.
        """
        self._print_step("THINK", "🧠", "Analyzing surplus-deficit pairs for rebalancing...")

        transfer_plan = []

        # Get all need locations (critical + low stock)
        needs = assessment["critical_skus"] + assessment["low_stock_skus"]
        surpluses = assessment["surplus_skus"]

        # Build surplus lookup: sku → list of surplus warehouses
        surplus_by_sku = {}
        for s in surpluses:
            sku = s["sku"]
            if sku not in surplus_by_sku:
                surplus_by_sku[sku] = []
            surplus_by_sku[sku].append(s)

        for need in needs:
            sku = need["sku"]
            need_wh = need["warehouse_id"]
            deficit = need["deficit"]

            if deficit <= 0:
                continue

            if sku not in surplus_by_sku or not surplus_by_sku[sku]:
                self._log_decision(
                    phase="THINK",
                    action="no_source_found",
                    details={"sku": sku, "warehouse": need_wh, "deficit": deficit},
                    reasoning=f"No surplus warehouse found for {sku}. External procurement needed.",
                )
                continue

            # Find best source (highest surplus)
            sources = sorted(surplus_by_sku[sku], key=lambda x: x["surplus"], reverse=True)
            source = sources[0]

            # Calculate transfer quantity
            transfer_qty = min(
                deficit + REORDER_THRESHOLD,  # Fill to above threshold
                source["surplus"],             # Don't over-drain source
                30,                            # Max 30 units per transfer
            )

            if transfer_qty <= 0:
                continue

            # Get forecast to validate (if available)
            forecast = self.toolkit.get_forecast(sku, need.get("pincode", ""))
            predicted_demand = forecast.get("predicted_total_demand", deficit)

            # Adjust transfer based on forecast
            if predicted_demand > transfer_qty:
                transfer_qty = min(transfer_qty + 5, source["surplus"])

            transfer_plan.append({
                "sku": sku,
                "from_warehouse": source["warehouse_id"],
                "to_warehouse": need_wh,
                "quantity": transfer_qty,
                "reason": f"Forecasted demand spike: {predicted_demand} units/24h. "
                          f"Current stock at {need_wh}: {need['current_stock']} "
                          f"(threshold: {need['reorder_threshold']}). "
                          f"Source {source['warehouse_id']} has surplus of {source['surplus']} units.",
                "priority": "high" if need["status"] == "critical" else "medium",
                "forecast": forecast,
            })

            # Update surplus tracking (so we don't over-allocate)
            source["surplus"] -= transfer_qty

        self._log_decision(
            phase="THINK",
            action="transfer_plan",
            details={
                "planned_transfers": len(transfer_plan),
                "total_units": sum(t["quantity"] for t in transfer_plan),
            },
            reasoning=f"Planned {len(transfer_plan)} transfers based on surplus-deficit matching "
                      f"and demand forecasts.",
        )

        print(f"      Planned {len(transfer_plan)} transfers")
        print(f"      Total units to move: {sum(t['quantity'] for t in transfer_plan)}")

        return transfer_plan

    def act_on_transfers(self, transfer_plan):
        """
        ACT: Execute the planned transfers.
        """
        self._print_step("ACT", "⚡", f"Executing {len(transfer_plan)} planned transfers...")

        results = []

        for plan in transfer_plan:
            result = self.toolkit.transfer_stock(
                from_warehouse=plan["from_warehouse"],
                to_warehouse=plan["to_warehouse"],
                sku=plan["sku"],
                quantity=plan["quantity"],
            )

            emoji = "✅" if result["success"] else "❌"
            print(f"      {emoji} Transfer {plan['quantity']} × {plan['sku']}: "
                  f"{plan['from_warehouse']} → {plan['to_warehouse']} "
                  f"({plan['priority']} priority)")

            if result["success"]:
                print(f"         Reason: {plan['reason'][:80]}...")

            self._log_decision(
                phase="ACT",
                action="execute_transfer",
                details={
                    "sku": plan["sku"],
                    "from": plan["from_warehouse"],
                    "to": plan["to_warehouse"],
                    "quantity": plan["quantity"],
                    "success": result["success"],
                    "priority": plan["priority"],
                },
                reasoning=plan["reason"],
            )

            results.append({**plan, **result})

        return results

    def reflect_on_actions(self, results):
        """
        REFLECT: Evaluate the impact of transfers.
        """
        self._print_step("REFLECT", "📊", "Evaluating rebalancing impact...")

        successful = [r for r in results if r.get("success", False)]
        failed = [r for r in results if not r.get("success", False)]

        # Check post-transfer state
        post_summary = self.toolkit.get_inventory_summary()

        reflection = {
            "total_transfers_attempted": len(results),
            "successful_transfers": len(successful),
            "failed_transfers": len(failed),
            "total_units_moved": sum(r["quantity"] for r in successful),
            "estimated_cost": sum(r["quantity"] * TRANSFER_COST_PER_UNIT for r in successful),
            "post_transfer_summary": post_summary.to_dict("records"),
            "sla_impact": "SLA compliance improved by reducing stockout risk at deficit warehouses",
        }

        self._log_decision(
            phase="REFLECT",
            action="post_transfer_assessment",
            details=reflection,
            reasoning=f"Completed {len(successful)} of {len(results)} planned transfers. "
                      f"Moved {reflection['total_units_moved']} total units at estimated cost of "
                      f"₹{reflection['estimated_cost']:.0f}. "
                      f"Failed transfers: {len(failed)} (insufficient stock or capacity constraints).",
        )

        print(f"\n      📊 Rebalancing Summary:")
        print(f"         Successful: {len(successful)} / {len(results)}")
        print(f"         Units moved: {reflection['total_units_moved']}")
        print(f"         Estimated cost: ₹{reflection['estimated_cost']:.0f}")

        return reflection

    def run_orchestration_cycle(self):
        """
        Run a complete orchestration cycle: OBSERVE → THINK → ACT → REFLECT.

        Returns:
            Complete orchestration report
        """
        print("\n" + "=" * 60)
        print("🚀 AGENTIC ORCHESTRATION: Starting Rebalancing Cycle")
        print("=" * 60)
        print(f"   Timestamp: {datetime.now().isoformat()}")
        print(f"   Warehouses: {', '.join(WAREHOUSE_IDS)}")

        # Phase 1: OBSERVE
        assessment = self.observe_inventory_state()

        # Phase 2: THINK
        transfer_plan = self.think_about_transfers(assessment)

        if not transfer_plan:
            print("\n   ℹ️  No transfers needed — inventory is balanced!")
            self._log_decision(
                phase="REFLECT",
                action="no_action_needed",
                details={"assessment": "balanced"},
                reasoning="All warehouses have adequate stock levels. No rebalancing required.",
            )
            return {
                "status": "balanced",
                "decision_log": self.decision_log,
                "transfers": [],
            }

        # Phase 3: ACT
        results = self.act_on_transfers(transfer_plan)

        # Phase 4: REFLECT
        reflection = self.reflect_on_actions(results)

        # Save decision log
        self._save_decision_log()

        print("\n" + "=" * 60)
        print("✅ ORCHESTRATION CYCLE COMPLETE")
        print("=" * 60)

        return {
            "status": "completed",
            "decision_log": self.decision_log,
            "transfers": results,
            "reflection": reflection,
        }

    def _save_decision_log(self):
        """Save the decision log to file."""
        LOGS_DIR.mkdir(parents=True, exist_ok=True)
        log_path = LOGS_DIR / f"orchestration_log_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"

        with open(log_path, "w") as f:
            json.dump(self.decision_log, f, indent=2, default=str)

        print(f"\n   💾 Decision log saved to {log_path}")

    def get_formatted_log(self) -> str:
        """Get a human-readable formatted decision log."""
        lines = []
        lines.append("=" * 70)
        lines.append("📋 INVENTORY ORCHESTRATION AGENT — Decision Log")
        lines.append("=" * 70)

        for entry in self.decision_log:
            lines.append(f"\n[Step {entry['step']}] {entry['phase']} — {entry['action']}")
            lines.append(f"  Time: {entry['timestamp']}")
            lines.append(f"  Reasoning: {entry['reasoning']}")
            if isinstance(entry['details'], dict):
                for k, v in entry['details'].items():
                    if k not in ('post_transfer_summary',):
                        lines.append(f"  {k}: {v}")
            lines.append("-" * 50)

        return "\n".join(lines)


def demo_orchestration():
    """Demo the orchestration agent."""
    agent = InventoryOrchestrationAgent()
    report = agent.run_orchestration_cycle()

    print("\n\n" + agent.get_formatted_log())

    return agent, report


if __name__ == "__main__":
    demo_orchestration()
