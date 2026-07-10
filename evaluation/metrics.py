"""
╔══════════════════════════════════════════════════════════════╗
║  📊 EVALUATION AGENT — Metrics Module                        ║
║  Precision@K, NDCG@K, MAPE, RMSE, SLA fulfillment rate      ║
╚══════════════════════════════════════════════════════════════╝
"""

import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).parent.parent))
from config import EVAL_K_VALUES


# ─── Recommendation Metrics ──────────────────────────────────

def precision_at_k(recommended_categories, relevant_category, k):
    """
    Precision@K: fraction of top-K recommendations that are relevant.

    Here, "relevant" means same masterCategory as the query product.

    Args:
        recommended_categories: list of categories for recommended items
        relevant_category: the target category
        k: number of top results to consider

    Returns:
        float: precision score
    """
    top_k = recommended_categories[:k]
    relevant = sum(1 for cat in top_k if cat == relevant_category)
    return relevant / k


def ndcg_at_k(recommended_categories, relevant_category, k):
    """
    Normalized Discounted Cumulative Gain (NDCG@K).

    Uses binary relevance: 1 if category matches, 0 otherwise.

    Args:
        recommended_categories: list of categories for recommended items
        relevant_category: the target category
        k: number of top results to consider

    Returns:
        float: NDCG score
    """
    top_k = recommended_categories[:k]

    # DCG
    dcg = 0.0
    for i, cat in enumerate(top_k):
        relevance = 1.0 if cat == relevant_category else 0.0
        dcg += relevance / np.log2(i + 2)  # i+2 because log2(1) = 0

    # Ideal DCG (all relevant items at top)
    n_relevant = sum(1 for cat in recommended_categories if cat == relevant_category)
    ideal_n = min(n_relevant, k)
    idcg = sum(1.0 / np.log2(i + 2) for i in range(ideal_n))

    return dcg / idcg if idcg > 0 else 0.0


def evaluate_recommendations(engine, catalog, n_queries=50, k_values=EVAL_K_VALUES):
    """
    Evaluate the recommendation engine on random catalog queries.

    For each query product, we use its description to find similar products
    and check if they share the same masterCategory.

    Returns:
        dict with metrics for each K value
    """
    print("=" * 60)
    print("📊 EVALUATION: Recommendation Quality")
    print("=" * 60)

    np.random.seed(42)
    sample_indices = np.random.choice(len(catalog), size=min(n_queries, len(catalog)), replace=False)

    results = {k: {"precision": [], "ndcg": []} for k in k_values}

    for idx in sample_indices:
        product = catalog.iloc[idx]
        query_category = product["masterCategory"]

        try:
            # Use product description as text query
            recs = engine.recommend_by_text(
                query=product["productDisplayName"],
                top_k=max(k_values),
            )

            rec_categories = [r["master_category"] for r in recs]

            for k in k_values:
                p = precision_at_k(rec_categories, query_category, k)
                n = ndcg_at_k(rec_categories, query_category, k)
                results[k]["precision"].append(p)
                results[k]["ndcg"].append(n)

        except Exception:
            continue

    # Aggregate
    metrics = {}
    for k in k_values:
        if results[k]["precision"]:
            metrics[f"precision@{k}"] = round(np.mean(results[k]["precision"]), 4)
            metrics[f"ndcg@{k}"] = round(np.mean(results[k]["ndcg"]), 4)

    print("\n   📊 Results:")
    for metric, value in metrics.items():
        print(f"      {metric}: {value:.4f}")

    return metrics


# ─── Forecasting Metrics ─────────────────────────────────────

def mape(actual, predicted):
    """Mean Absolute Percentage Error."""
    actual = np.array(actual, dtype=float)
    predicted = np.array(predicted, dtype=float)
    nonzero = actual > 0
    if nonzero.sum() == 0:
        return 0.0
    return float(np.mean(np.abs((actual[nonzero] - predicted[nonzero]) / actual[nonzero])) * 100)


def rmse(actual, predicted):
    """Root Mean Squared Error."""
    actual = np.array(actual, dtype=float)
    predicted = np.array(predicted, dtype=float)
    return float(np.sqrt(np.mean((actual - predicted) ** 2)))


def evaluate_forecasts(forecaster, n_skus=5, n_pincodes=3):
    """
    Evaluate forecast quality using held-out test data.

    Returns:
        dict with MAPE and RMSE metrics
    """
    print("\n" + "=" * 60)
    print("📊 EVALUATION: Forecast Accuracy")
    print("=" * 60)

    from config import FORECASTS_DIR

    metrics_path = FORECASTS_DIR / "forecast_metrics.csv"
    if metrics_path.exists():
        metrics_df = pd.read_csv(metrics_path)
        avg_mape = metrics_df["mape"].mean()
        avg_rmse = metrics_df["rmse"].mean()
    else:
        # Run forecasts to get metrics
        _, metrics_df = forecaster.forecast_all(n_skus=n_skus, n_pincodes=n_pincodes)
        avg_mape = metrics_df["mape"].mean() if len(metrics_df) > 0 else 0
        avg_rmse = metrics_df["rmse"].mean() if len(metrics_df) > 0 else 0

    result = {
        "avg_mape": round(avg_mape, 2),
        "avg_rmse": round(avg_rmse, 2),
        "n_models": len(metrics_df),
    }

    print(f"\n   📊 Results:")
    print(f"      Average MAPE: {result['avg_mape']:.2f}%")
    print(f"      Average RMSE: {result['avg_rmse']:.2f}")
    print(f"      Models evaluated: {result['n_models']}")

    return result


# ─── SLA Fulfillment ─────────────────────────────────────────

def evaluate_sla_fulfillment(toolkit, n_simulations=100):
    """
    Simulate orders and measure SLA (60-min delivery) fulfillment rate.

    An order is fulfilled if the warehouse has stock for the requested SKU.
    """
    print("\n" + "=" * 60)
    print("📊 EVALUATION: SLA Fulfillment Rate")
    print("=" * 60)

    np.random.seed(42)

    inventory = toolkit.inventory
    skus = inventory["sku"].unique()
    warehouses = inventory["warehouse_id"].unique()

    fulfilled = 0
    total = 0

    for _ in range(n_simulations):
        sku = np.random.choice(skus)
        wh = np.random.choice(warehouses)

        stock = toolkit.check_stock(wh, sku)
        total += 1
        if stock["current_stock"] > 0:
            fulfilled += 1

    rate = fulfilled / total if total > 0 else 0

    result = {
        "fulfillment_rate": round(rate * 100, 2),
        "fulfilled_orders": fulfilled,
        "total_orders": total,
    }

    print(f"\n   📊 Results:")
    print(f"      SLA Fulfillment Rate: {result['fulfillment_rate']:.1f}%")
    print(f"      Fulfilled: {fulfilled} / {total}")

    return result


# ─── Full Evaluation ─────────────────────────────────────────

def run_full_evaluation(engine=None, forecaster=None, toolkit=None):
    """Run all evaluation metrics."""
    print("\n" + "🏆" * 30)
    print("   FULL EVALUATION REPORT — Zintoo AI Platform")
    print("🏆" * 30)

    report = {}

    # Recommendation evaluation
    if engine is not None:
        report["recommendation"] = evaluate_recommendations(
            engine, engine.catalog, n_queries=50
        )

    # Forecast evaluation
    if forecaster is not None:
        report["forecasting"] = evaluate_forecasts(forecaster)

    # SLA evaluation
    if toolkit is not None:
        report["sla"] = evaluate_sla_fulfillment(toolkit)

    print("\n\n" + "=" * 60)
    print("📊 EVALUATION SUMMARY")
    print("=" * 60)
    for module, metrics in report.items():
        print(f"\n   [{module.upper()}]")
        for key, value in metrics.items():
            print(f"      {key}: {value}")

    return report


if __name__ == "__main__":
    print("Run this module through the main pipeline or Streamlit dashboard.")
