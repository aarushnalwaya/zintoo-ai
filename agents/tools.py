"""
╔══════════════════════════════════════════════════════════════╗
║  🤖 BACKEND AGENT — Agent Tools                             ║
║  Inventory management tools for the orchestration agent      ║
╚══════════════════════════════════════════════════════════════╝
"""

import sys
from pathlib import Path
from datetime import datetime

import pandas as pd

sys.path.insert(0, str(Path(__file__).parent.parent))
from config import DATA_DIR, WAREHOUSE_PINCODE_MAP


class InventoryToolkit:
    """
    Tools for the inventory orchestration agent.
    Provides functions to check stock, transfer inventory, and get forecasts.
    """

    def __init__(self, inventory_df=None, forecaster=None):
        if inventory_df is not None:
            self.inventory = inventory_df.copy()
        else:
            inv_path = DATA_DIR / "warehouse_inventory.csv"
            if inv_path.exists():
                self.inventory = pd.read_csv(inv_path)
            else:
                raise FileNotFoundError(f"Inventory data not found at {inv_path}")

        self.forecaster = forecaster
        self.transfer_log = []
        self.warehouse_pincode_map = WAREHOUSE_PINCODE_MAP

    def check_stock(self, warehouse_id: str, sku: str) -> dict:
        """
        Check current stock level for a SKU at a specific warehouse.

        Returns:
            dict with stock info
        """
        mask = (self.inventory["warehouse_id"] == warehouse_id) & (self.inventory["sku"] == sku)
        match = self.inventory[mask]

        if len(match) == 0:
            return {
                "warehouse_id": warehouse_id,
                "sku": sku,
                "status": "not_found",
                "current_stock": 0,
                "message": f"SKU {sku} not found in warehouse {warehouse_id}",
            }

        row = match.iloc[0]
        stock = int(row["current_stock"])
        threshold = int(row["reorder_threshold"])

        return {
            "warehouse_id": warehouse_id,
            "sku": sku,
            "pincode": row.get("pincode", ""),
            "current_stock": stock,
            "reorder_threshold": threshold,
            "max_capacity": int(row.get("max_capacity", 100)),
            "status": "critical" if stock <= threshold // 2 else "low" if stock <= threshold else "healthy",
            "needs_restock": stock <= threshold,
        }

    def get_all_warehouses_for_sku(self, sku: str) -> list:
        """Get stock levels across all warehouses for a SKU."""
        mask = self.inventory["sku"] == sku
        matches = self.inventory[mask]

        results = []
        for _, row in matches.iterrows():
            stock = int(row["current_stock"])
            threshold = int(row["reorder_threshold"])
            results.append({
                "warehouse_id": row["warehouse_id"],
                "pincode": row.get("pincode", ""),
                "current_stock": stock,
                "reorder_threshold": threshold,
                "surplus": max(0, stock - threshold * 2),
                "deficit": max(0, threshold - stock),
                "status": "critical" if stock <= threshold // 2 else "low" if stock <= threshold else "healthy",
            })

        return sorted(results, key=lambda x: x["current_stock"], reverse=True)

    def transfer_stock(self, from_warehouse: str, to_warehouse: str, sku: str, quantity: int) -> dict:
        """
        Transfer stock between warehouses.

        Args:
            from_warehouse: Source warehouse ID
            to_warehouse: Destination warehouse ID
            sku: SKU to transfer
            quantity: Number of units to transfer

        Returns:
            dict with transfer result
        """
        # Validate source
        source = self.check_stock(from_warehouse, sku)
        if source["status"] == "not_found":
            return {"success": False, "error": f"SKU {sku} not found in {from_warehouse}"}

        if source["current_stock"] < quantity:
            return {
                "success": False,
                "error": f"Insufficient stock in {from_warehouse}: has {source['current_stock']}, need {quantity}",
            }

        # Validate destination
        dest = self.check_stock(to_warehouse, sku)
        if dest["status"] == "not_found":
            return {"success": False, "error": f"SKU {sku} slot not found in {to_warehouse}"}

        if dest["current_stock"] + quantity > dest.get("max_capacity", 100):
            return {
                "success": False,
                "error": f"Would exceed capacity at {to_warehouse}: has {dest['current_stock']}, adding {quantity}, max={dest.get('max_capacity', 100)}",
            }

        # Execute transfer
        src_mask = (self.inventory["warehouse_id"] == from_warehouse) & (self.inventory["sku"] == sku)
        dst_mask = (self.inventory["warehouse_id"] == to_warehouse) & (self.inventory["sku"] == sku)

        self.inventory.loc[src_mask, "current_stock"] -= quantity
        self.inventory.loc[dst_mask, "current_stock"] += quantity

        # Log the transfer
        transfer_record = {
            "timestamp": datetime.now().isoformat(),
            "from_warehouse": from_warehouse,
            "to_warehouse": to_warehouse,
            "sku": sku,
            "quantity": quantity,
            "from_stock_before": source["current_stock"],
            "from_stock_after": source["current_stock"] - quantity,
            "to_stock_before": dest["current_stock"],
            "to_stock_after": dest["current_stock"] + quantity,
            "reason": "rebalance",
        }
        self.transfer_log.append(transfer_record)

        return {
            "success": True,
            "transfer": transfer_record,
            "message": f"Transferred {quantity} units of {sku} from {from_warehouse} to {to_warehouse}",
        }

    def get_forecast(self, sku: str, pincode: str, hours: int = 24) -> dict:
        """Get demand forecast for a SKU at a pincode."""
        if self.forecaster is None:
            # Return a simple estimate based on historical average
            mask = (self.inventory["sku"] == sku) & (self.inventory["pincode"] == pincode)
            avg_stock = self.inventory[mask]["current_stock"].mean() if len(self.inventory[mask]) > 0 else 10

            return {
                "sku": sku,
                "pincode": pincode,
                "hours": hours,
                "predicted_total_demand": int(avg_stock * 0.3),  # ~30% of stock as daily demand
                "peak_hour_demand": int(avg_stock * 0.05),
                "source": "estimate",
            }

        try:
            forecast_df, metrics = self.forecaster.get_demand_prediction(sku, pincode, hours)
            return {
                "sku": sku,
                "pincode": pincode,
                "hours": hours,
                "predicted_total_demand": int(forecast_df["predicted_demand"].sum()),
                "peak_hour_demand": int(forecast_df["predicted_demand"].max()),
                "peak_hour": str(forecast_df.loc[forecast_df["predicted_demand"].idxmax(), "timestamp"]),
                "hourly_forecast": forecast_df.to_dict("records"),
                "metrics": metrics,
                "source": "prophet",
            }
        except Exception as e:
            return {
                "sku": sku,
                "pincode": pincode,
                "error": str(e),
                "source": "failed",
            }

    def get_transfer_log(self) -> list:
        """Get all transfer records."""
        return self.transfer_log

    def get_inventory_summary(self) -> pd.DataFrame:
        """Get a summary of inventory across all warehouses."""
        summary = self.inventory.groupby("warehouse_id").agg(
            total_skus=("sku", "nunique"),
            total_stock=("current_stock", "sum"),
            avg_stock=("current_stock", "mean"),
            min_stock=("current_stock", "min"),
            low_stock_skus=("current_stock", lambda x: (x <= 10).sum()),
        ).reset_index()
        return summary
