"""
╔══════════════════════════════════════════════════════════════╗
║  🎨 FRONTEND AGENT — Streamlit Dashboard                    ║
║  Interactive demo for all three Zintoo AI modules            ║
╚══════════════════════════════════════════════════════════════╝

Run with:
  cd zintoo && streamlit run dashboard/app.py
"""

import sys
import os
from pathlib import Path

# Ensure project root is on path
PROJECT_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(PROJECT_ROOT))
os.chdir(str(PROJECT_ROOT))

import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots
from PIL import Image
from datetime import datetime, timedelta

from config import (
    DATASET_DIR, IMAGES_DIR, DATA_DIR, FAISS_INDEX_PATH,
    PIN_CODES, WAREHOUSE_IDS, FORECASTS_DIR,
)

# ─── Page Config ──────────────────────────────────────────────

st.set_page_config(
    page_title="Zintoo — AI Fashion Intelligence",
    page_icon="👗",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ─── Custom CSS ───────────────────────────────────────────────

st.markdown("""
<style>
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700&display=swap');

    .main { font-family: 'Inter', sans-serif; }

    .stApp {
        background: linear-gradient(135deg, #0f0c29 0%, #1a1a2e 50%, #16213e 100%);
    }

    .hero-title {
        font-size: 3rem;
        font-weight: 700;
        background: linear-gradient(120deg, #f093fb 0%, #f5576c 50%, #4facfe 100%);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        text-align: center;
        margin-bottom: 0.5rem;
    }

    .hero-subtitle {
        font-size: 1.1rem;
        color: #a0a0b0;
        text-align: center;
        margin-bottom: 2rem;
    }

    .product-card {
        background: rgba(255, 255, 255, 0.05);
        border: 1px solid rgba(255, 255, 255, 0.1);
        border-radius: 16px;
        padding: 12px;
        margin: 8px 0;
        backdrop-filter: blur(10px);
        transition: transform 0.2s, box-shadow 0.2s;
    }

    .product-card:hover {
        transform: translateY(-4px);
        box-shadow: 0 8px 32px rgba(240, 147, 251, 0.15);
    }

    .score-badge {
        display: inline-block;
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        color: white;
        padding: 4px 12px;
        border-radius: 20px;
        font-size: 0.85rem;
        font-weight: 600;
    }

    .metric-card {
        background: rgba(255, 255, 255, 0.05);
        border: 1px solid rgba(255, 255, 255, 0.1);
        border-radius: 12px;
        padding: 20px;
        text-align: center;
    }

    .agent-step {
        background: rgba(255, 255, 255, 0.03);
        border-left: 3px solid #f093fb;
        padding: 12px 16px;
        margin: 8px 0;
        border-radius: 0 8px 8px 0;
    }

    .status-healthy { color: #2ecc71; }
    .status-low { color: #f39c12; }
    .status-critical { color: #e74c3c; }

    /* Hide Streamlit branding */
    #MainMenu {visibility: hidden;}
    footer {visibility: hidden;}
</style>
""", unsafe_allow_html=True)


# ─── Cached Resource Loaders ─────────────────────────────────

@st.cache_resource
def load_recommendation_engine():
    """Load the recommendation engine (cached)."""
    try:
        from models.recommendation import RecommendationEngine
        return RecommendationEngine()
    except Exception as e:
        st.error(f"⚠️ Recommendation engine not available: {e}")
        return None


@st.cache_resource
def load_forecaster():
    """Load the demand forecaster (cached)."""
    try:
        from models.demand_forecast import DemandForecaster
        return DemandForecaster()
    except Exception as e:
        st.error(f"⚠️ Forecaster not available: {e}")
        return None


@st.cache_data
def load_catalog():
    """Load the product catalog."""
    catalog_path = DATASET_DIR / "catalog.csv"
    if catalog_path.exists():
        return pd.read_csv(catalog_path)
    return None


@st.cache_data
def load_inventory():
    """Load warehouse inventory."""
    inv_path = DATA_DIR / "warehouse_inventory.csv"
    if inv_path.exists():
        return pd.read_csv(inv_path)
    return None


@st.cache_data
def load_demand_data():
    """Load demand history."""
    demand_path = DATA_DIR / "demand_history.csv"
    if demand_path.exists():
        return pd.read_csv(demand_path, parse_dates=["timestamp"])
    return None


# ─── Helper Functions ─────────────────────────────────────────

def render_product_card(product, score=None):
    """Render a product card with image and details."""
    img_path = product.get("image_path", "")
    name = product.get("name", product.get("productDisplayName", "Unknown"))
    category = product.get("master_category", product.get("masterCategory", ""))
    color = product.get("color", product.get("baseColour", ""))
    article = product.get("article_type", product.get("articleType", ""))

    col1, col2 = st.columns([1, 2])

    with col1:
        if img_path and Path(img_path).exists():
            img = Image.open(img_path)
            st.image(img, width=120)
        else:
            st.write("🖼️ No image")

    with col2:
        st.markdown(f"**{name}**")
        if score is not None:
            st.markdown(f'<span class="score-badge">Score: {score:.4f}</span>', unsafe_allow_html=True)
        st.caption(f"📦 {category} → {article} | 🎨 {color}")


# ─── Hero Header ──────────────────────────────────────────────

st.markdown('<h1 class="hero-title">👗 Zintoo</h1>', unsafe_allow_html=True)
st.markdown('<p class="hero-subtitle">AI-Powered Hyper-Local Fashion Intelligence Platform</p>', unsafe_allow_html=True)

# ─── Sidebar ──────────────────────────────────────────────────

with st.sidebar:
    st.image("https://img.icons8.com/fluency/96/artificial-intelligence.png", width=64)
    st.title("Navigation")

    # System status
    st.markdown("### 📡 System Status")

    faiss_ok = FAISS_INDEX_PATH.exists()
    demand_ok = (DATA_DIR / "demand_history.csv").exists()
    inv_ok = (DATA_DIR / "warehouse_inventory.csv").exists()

    st.markdown(f"{'✅' if faiss_ok else '❌'} Recommendation Engine")
    st.markdown(f"{'✅' if demand_ok else '❌'} Demand Forecasting")
    st.markdown(f"{'✅' if inv_ok else '❌'} Inventory Orchestration")

    st.divider()
    st.markdown("### ℹ️ About")
    st.markdown(
        "Zintoo is a quick-commerce fashion platform with "
        "60-minute delivery and AI-powered personalization."
    )

# ─── Main Tabs ────────────────────────────────────────────────

tab1, tab2, tab3, tab4 = st.tabs([
    "🛍️ Recommendations",
    "📈 Demand Forecast",
    "🤖 Inventory Agent",
    "📊 Evaluation",
])

# ═══════════════════════════════════════════════════════════════
# TAB 1: RECOMMENDATIONS
# ═══════════════════════════════════════════════════════════════

with tab1:
    st.markdown("## 🛍️ Multimodal Fashion Recommendations")
    st.markdown("Enter a text description, upload an image, or both!")

    col_input, col_config = st.columns([3, 1])

    with col_input:
        text_query = st.text_input(
            "🔍 Describe what you're looking for:",
            placeholder="casual kurta for a college fest",
            key="rec_query",
        )

        uploaded_image = st.file_uploader(
            "📸 Or upload a fashion image:",
            type=["jpg", "jpeg", "png"],
            key="rec_image",
        )

    with col_config:
        top_k = st.slider("Results to show:", 3, 20, 5, key="rec_topk")
        alpha = st.slider("Text vs Image weight:", 0.0, 1.0, 0.5, key="rec_alpha",
                         help="1.0 = text only, 0.0 = image only")

        gender_filter = st.selectbox("Gender filter:", ["All", "Men", "Women", "Unisex", "Boys", "Girls"])
        category_filter = st.selectbox("Category filter:", ["All", "Apparel", "Footwear", "Accessories"])

    if st.button("🎯 Get Recommendations", type="primary", key="rec_btn"):
        engine = load_recommendation_engine()

        if engine is None:
            st.error("⚠️ Recommendation engine not loaded. Please build the FAISS index first.")
        else:
            filters = {}
            if gender_filter != "All":
                filters["gender"] = gender_filter
            if category_filter != "All":
                filters["masterCategory"] = category_filter

            with st.spinner("🔍 Finding the perfect matches..."):
                try:
                    if uploaded_image and text_query:
                        # Multimodal
                        image = Image.open(uploaded_image).convert("RGB")
                        results = engine.recommend_multimodal(
                            text=text_query, image_input=image,
                            alpha=alpha, top_k=top_k,
                            filters=filters or None,
                        )
                        mode = "🔀 Multimodal"
                    elif uploaded_image:
                        # Image only
                        image = Image.open(uploaded_image).convert("RGB")
                        results = engine.recommend_by_image(
                            image_input=image, top_k=top_k,
                            filters=filters or None,
                        )
                        mode = "🖼️ Image"
                    elif text_query:
                        # Text only
                        results = engine.recommend_by_text(
                            query=text_query, top_k=top_k,
                            filters=filters or None,
                        )
                        mode = "📝 Text"
                    else:
                        st.warning("Please enter a text query or upload an image.")
                        results = []
                        mode = ""

                    if results:
                        st.success(f"{mode} Search — Found {len(results)} results")

                        # Display results in a grid
                        cols = st.columns(min(3, len(results)))
                        for i, result in enumerate(results):
                            with cols[i % 3]:
                                with st.container():
                                    img_path = result.get("image_path", "")
                                    if img_path and Path(img_path).exists():
                                        st.image(Image.open(img_path), use_container_width=True)

                                    st.markdown(f"**{result['name'][:50]}**")
                                    st.markdown(
                                        f'<span class="score-badge">{result["similarity_score"]:.4f}</span>',
                                        unsafe_allow_html=True,
                                    )
                                    st.caption(
                                        f"{result['master_category']} → {result['article_type']} | "
                                        f"🎨 {result['color']} | {result['gender']}"
                                    )
                                    st.divider()

                except Exception as e:
                    st.error(f"Error: {e}")


# ═══════════════════════════════════════════════════════════════
# TAB 2: DEMAND FORECAST
# ═══════════════════════════════════════════════════════════════

with tab2:
    st.markdown("## 📈 Hyper-Local Demand Forecasting")
    st.markdown("Predict hourly SKU-level demand for specific pin codes")

    demand_data = load_demand_data()

    if demand_data is not None:
        col1, col2, col3 = st.columns(3)

        with col1:
            available_skus = sorted(demand_data["sku"].unique().tolist())
            selected_sku = st.selectbox("Select SKU:", available_skus[:20], key="fc_sku")

        with col2:
            available_pincodes = sorted(demand_data["pincode"].unique().tolist())
            selected_pincode = st.selectbox("Select Pin Code:", available_pincodes, key="fc_pin")

        with col3:
            forecast_hours = st.slider("Forecast horizon (hours):", 6, 72, 24, key="fc_hours")

        if st.button("🔮 Generate Forecast", type="primary", key="fc_btn"):
            with st.spinner("Training Prophet model..."):
                try:
                    forecaster = load_forecaster()
                    if forecaster:
                        forecast_df, model, metrics = forecaster.train_forecast(
                            selected_sku, selected_pincode, forecast_hours
                        )

                        # Plot
                        historical = demand_data[
                            (demand_data["sku"] == selected_sku) &
                            (demand_data["pincode"] == selected_pincode)
                        ].tail(168)  # Last 7 days

                        fig = go.Figure()

                        # Historical
                        fig.add_trace(go.Scatter(
                            x=historical["timestamp"],
                            y=historical["demand"],
                            mode="lines",
                            name="Historical Demand",
                            line=dict(color="#3498db", width=1.5),
                        ))

                        # Forecast
                        fig.add_trace(go.Scatter(
                            x=forecast_df["timestamp"],
                            y=forecast_df["predicted_demand"],
                            mode="lines",
                            name="Forecast",
                            line=dict(color="#e74c3c", width=2, dash="dash"),
                        ))

                        # Confidence interval
                        fig.add_trace(go.Scatter(
                            x=pd.concat([forecast_df["timestamp"], forecast_df["timestamp"][::-1]]),
                            y=pd.concat([forecast_df["upper_bound"], forecast_df["lower_bound"][::-1]]),
                            fill="toself",
                            fillcolor="rgba(231, 76, 60, 0.15)",
                            line=dict(color="rgba(0,0,0,0)"),
                            name="95% Confidence",
                        ))

                        fig.update_layout(
                            title=f"Demand Forecast: {selected_sku} @ {selected_pincode}",
                            xaxis_title="Time",
                            yaxis_title="Demand (units/hour)",
                            template="plotly_dark",
                            height=500,
                            font=dict(family="Inter"),
                        )

                        st.plotly_chart(fig, use_container_width=True)

                        # Metrics
                        col_m1, col_m2, col_m3, col_m4 = st.columns(4)
                        with col_m1:
                            st.metric("Total Predicted Demand",
                                     f"{int(forecast_df['predicted_demand'].sum())} units")
                        with col_m2:
                            st.metric("Peak Hour Demand",
                                     f"{int(forecast_df['predicted_demand'].max())} units")
                        with col_m3:
                            if metrics:
                                st.metric("MAPE", f"{metrics.get('mape', 0):.1f}%")
                        with col_m4:
                            if metrics:
                                st.metric("RMSE", f"{metrics.get('rmse', 0):.2f}")

                        # Show forecast table
                        with st.expander("📋 Hourly Forecast Data"):
                            st.dataframe(forecast_df, use_container_width=True)

                except Exception as e:
                    st.error(f"Forecast error: {e}")

        # Historical demand overview
        if demand_data is not None:
            st.divider()
            st.markdown("### 📊 Historical Demand Overview")

            # Aggregate by hour of day
            demand_hourly = demand_data.groupby("hour")["demand"].mean().reset_index()
            fig_hourly = px.bar(
                demand_hourly, x="hour", y="demand",
                title="Average Demand by Hour of Day",
                labels={"hour": "Hour", "demand": "Avg Demand"},
                template="plotly_dark",
            )
            fig_hourly.update_traces(marker_color='#667eea')
            st.plotly_chart(fig_hourly, use_container_width=True)

    else:
        st.warning("⚠️ Demand data not found. Run generate_synthetic.py first.")


# ═══════════════════════════════════════════════════════════════
# TAB 3: INVENTORY ORCHESTRATION AGENT
# ═══════════════════════════════════════════════════════════════

with tab3:
    st.markdown("## 🤖 Agentic Inventory Orchestration")
    st.markdown("Autonomous stock rebalancing across micro-warehouses")

    inventory = load_inventory()

    if inventory is not None:
        # Current inventory heatmap
        st.markdown("### 📦 Current Inventory Levels")

        pivot = inventory.pivot_table(
            values="current_stock",
            index="sku",
            columns="warehouse_id",
            aggfunc="first",
        ).head(15)

        fig_heat = px.imshow(
            pivot,
            labels=dict(x="Warehouse", y="SKU", color="Stock"),
            color_continuous_scale="RdYlGn",
            title="Stock Levels by SKU × Warehouse",
            template="plotly_dark",
        )
        fig_heat.update_layout(height=500, font=dict(family="Inter"))
        st.plotly_chart(fig_heat, use_container_width=True)

        # Warehouse summary
        st.markdown("### 🏭 Warehouse Summary")
        summary = inventory.groupby("warehouse_id").agg(
            Total_SKUs=("sku", "nunique"),
            Total_Stock=("current_stock", "sum"),
            Avg_Stock=("current_stock", "mean"),
            Low_Stock=("current_stock", lambda x: (x <= 10).sum()),
        ).reset_index()
        summary["Avg_Stock"] = summary["Avg_Stock"].round(1)

        st.dataframe(summary, use_container_width=True, hide_index=True)

        # Run orchestration
        st.divider()
        st.markdown("### ⚡ Run Autonomous Rebalancing")

        if st.button("🚀 Start Orchestration Cycle", type="primary", key="orch_btn"):
            with st.spinner("🤖 Agent is analyzing inventory and executing transfers..."):
                try:
                    from agents.tools import InventoryToolkit
                    from agents.inventory_agent import InventoryOrchestrationAgent

                    toolkit = InventoryToolkit(inventory_df=inventory)
                    agent = InventoryOrchestrationAgent(toolkit=toolkit)
                    report = agent.run_orchestration_cycle()

                    st.success(f"✅ Orchestration complete! Status: {report['status']}")

                    # Decision log
                    st.markdown("### 📋 Agent Decision Log")
                    for entry in report["decision_log"]:
                        phase_emoji = {
                            "OBSERVE": "🔍",
                            "THINK": "🧠",
                            "ACT": "⚡",
                            "REFLECT": "📊",
                        }.get(entry["phase"], "➡️")

                        st.markdown(
                            f'<div class="agent-step">'
                            f'<strong>[Step {entry["step"]}] {phase_emoji} {entry["phase"]}</strong> — {entry["action"]}<br>'
                            f'<em>{entry["reasoning"][:150]}</em>'
                            f'</div>',
                            unsafe_allow_html=True,
                        )

                    # Transfer log
                    if report.get("transfers"):
                        st.markdown("### 🔄 Transfers Executed")
                        transfers = []
                        for t in report["transfers"]:
                            transfers.append({
                                "SKU": t.get("sku", ""),
                                "From": t.get("from_warehouse", ""),
                                "To": t.get("to_warehouse", ""),
                                "Quantity": t.get("quantity", 0),
                                "Priority": t.get("priority", ""),
                                "Success": "✅" if t.get("success", False) else "❌",
                            })
                        st.dataframe(pd.DataFrame(transfers), use_container_width=True, hide_index=True)

                    # Reflection summary
                    if report.get("reflection"):
                        ref = report["reflection"]
                        col1, col2, col3, col4 = st.columns(4)
                        with col1:
                            st.metric("Successful Transfers", ref.get("successful_transfers", 0))
                        with col2:
                            st.metric("Units Moved", ref.get("total_units_moved", 0))
                        with col3:
                            st.metric("Est. Cost", f"₹{ref.get('estimated_cost', 0):.0f}")
                        with col4:
                            st.metric("Failed", ref.get("failed_transfers", 0))

                except Exception as e:
                    st.error(f"Orchestration error: {e}")
                    import traceback
                    st.code(traceback.format_exc())

    else:
        st.warning("⚠️ Inventory data not found. Run generate_synthetic.py first.")


# ═══════════════════════════════════════════════════════════════
# TAB 4: EVALUATION
# ═══════════════════════════════════════════════════════════════

with tab4:
    st.markdown("## 📊 Evaluation Report")
    st.markdown("Quantitative metrics for each module")

    # Check what's available
    col1, col2, col3 = st.columns(3)

    with col1:
        st.markdown("### 🛍️ Recommendations")
        st.markdown("""
        **Metrics:**
        - Precision@K: % of top-K results sharing same category
        - NDCG@K: Normalized discounted cumulative gain

        **Method:** Query each product by its description,
        check if results match the same masterCategory.
        """)

        if st.button("Run Recommendation Eval", key="eval_rec"):
            engine = load_recommendation_engine()
            if engine:
                with st.spinner("Evaluating recommendations (50 queries)..."):
                    from evaluation.metrics import evaluate_recommendations
                    metrics = evaluate_recommendations(engine, engine.catalog, n_queries=50)
                    for k, v in metrics.items():
                        st.metric(k, f"{v:.4f}")
            else:
                st.warning("Engine not loaded")

    with col2:
        st.markdown("### 📈 Forecasting")
        st.markdown("""
        **Metrics:**
        - MAPE: Mean Absolute Percentage Error
        - RMSE: Root Mean Squared Error

        **Method:** Train/test split on demand history,
        evaluate on held-out 24-hour window.
        """)

        metrics_path = FORECASTS_DIR / "forecast_metrics.csv"
        if metrics_path.exists():
            metrics_df = pd.read_csv(metrics_path)
            st.metric("Avg MAPE", f"{metrics_df['mape'].mean():.1f}%")
            st.metric("Avg RMSE", f"{metrics_df['rmse'].mean():.2f}")
            st.metric("Models Evaluated", len(metrics_df))
        else:
            st.info("Run forecasts first to see metrics")

    with col3:
        st.markdown("### 🤖 SLA Fulfillment")
        st.markdown("""
        **Metrics:**
        - SLA Fulfillment Rate: % of orders that can be
          served from local warehouse stock

        **Method:** Simulate random orders and check
        stock availability.
        """)

        if inventory is not None:
            if st.button("Run SLA Eval", key="eval_sla"):
                with st.spinner("Simulating 100 orders..."):
                    from agents.tools import InventoryToolkit
                    from evaluation.metrics import evaluate_sla_fulfillment
                    toolkit = InventoryToolkit(inventory_df=inventory)
                    sla = evaluate_sla_fulfillment(toolkit)
                    st.metric("Fulfillment Rate", f"{sla['fulfillment_rate']:.1f}%")
                    st.metric("Fulfilled", f"{sla['fulfilled_orders']}/{sla['total_orders']}")

    # Design tradeoffs
    st.divider()
    st.markdown("### 🔧 Design Trade-offs")
    st.markdown("""
    | Decision | Choice | Rationale |
    |----------|--------|-----------|
    | **Embedding Model** | FashionCLIP | Fine-tuned on fashion data, outperforms generic CLIP |
    | **Vector Search** | FAISS IndexFlatIP | Exact search, fast for <100K items. Use IVF for millions |
    | **Forecasting** | Prophet | Handles multiple seasonalities, easy regressor addition |
    | **Agent** | Rule-based ReAct | No external LLM API key needed, deterministic, auditable |
    | **Dataset** | Fashion Product Images (Small) | 280MB fits Colab, 44K products sufficient for demo |
    | **Weather API** | Open-Meteo | Free, no API key, covers Indian cities |
    """)


# ─── Footer ──────────────────────────────────────────────────

st.divider()
st.markdown(
    '<p style="text-align:center; color:#666; font-size:0.85rem;">'
    '🚀 Built for PS6 — Zintoo: AI-Powered Hyper-Local Fashion Intelligence Platform'
    '</p>',
    unsafe_allow_html=True,
)
