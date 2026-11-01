import json, datetime
from pathlib import Path
import joblib, pandas as pd, streamlit as st
from src.nlp import ItemMatcher, process_text
from src.optimize import aggregate, optimize

st.set_page_config(page_title="AI Grocery Optimizer", page_icon="🛒", layout="wide")
LOG = Path("data/query_log.csv")


@st.cache_resource
def load_assets():
    prices = pd.read_csv("data/prices.csv")
    unit_map = prices.drop_duplicates("product").set_index("product")["unit"].to_dict()
    try:
        matcher = joblib.load("models/matcher.joblib")
    except Exception:  # e.g. library version mismatch -> rebuild quickly
        matcher = ItemMatcher(sorted(unit_map))
    meta = json.load(open("models/metrics.json"))
    return prices, unit_map, matcher, meta


def log_query(text, n_items, n_unmatched):
    LOG.parent.mkdir(exist_ok=True)
    row = pd.DataFrame([{"time": datetime.datetime.now().isoformat(timespec="seconds"),
                         "text": text, "n_items": n_items, "n_unmatched": n_unmatched}])
    row.to_csv(LOG, mode="a", header=not LOG.exists(), index=False)


prices, unit_map, matcher, meta = load_assets()

st.title("🛒 AI-Based Grocery Optimization System")
st.caption("Type your grocery list in plain English. NLP understands it, the optimizer finds the cheapest way to buy it.")

tab1, tab2, tab3 = st.tabs(["🧠 Optimizer", "⚙️ MLOps: Model Info", "📊 Monitoring"])

with tab1:
    text = st.text_area("Your grocery list", "2 kg rice, 1 litre milk, half kg tomatoes, 6 eggs and some sugar", height=110)
    if st.button("Optimize my basket", type="primary"):
        items = process_text(text, matcher, unit_map)
        if not items:
            st.warning("I couldn't find any items. Try: 2 kg rice, 1 litre milk")
        else:
            df = pd.DataFrame(items)
            n_unmatched = int(df["product"].isna().sum())
            log_query(text, len(df), n_unmatched)

            st.subheader("1. What the NLP model understood")
            st.dataframe(df)
            if n_unmatched:
                st.warning(f"{n_unmatched} item(s) were not recognised and are skipped.")

            basket = aggregate(items)
            if basket:
                res = optimize(basket, prices)
                st.subheader("2. Total cost if you buy from ONE store (incl. delivery)")
                st.bar_chart(pd.Series(res["single_totals"], name="Total (INR)"))

                st.subheader("3. Best plan")
                c1, c2, c3 = st.columns(3)
                c1.metric("Best single store", res["best_single_store"], f"₹{res['best_single_total']}")
                c2.metric("Split across stores", f"{len(res['stores_used'])} stores", f"₹{res['split_total']}")
                best = min(res["best_single_total"], res["split_total"])
                c3.metric("You save vs costliest store", f"₹{round(res['worst_single_total'] - best, 2)}")

                if res["split_total"] < res["best_single_total"]:
                    st.success("Cheapest: buy each item from its cheapest store (table below).")
                else:
                    st.success(f"Cheapest: buy everything from {res['best_single_store']}.")
                st.dataframe(res["split_plan"])

with tab2:
    st.subheader("Current production model")
    c1, c2, c3 = st.columns(3)
    c1.metric("Version", meta["model_version"])
    c2.metric("Accuracy on typo test set", f"{meta['accuracy'] * 100:.1f}%")
    c3.metric("Products in catalog", meta["n_products"])
    st.json(meta)
    st.markdown("**Pipeline:** GitHub push → GitHub Actions retrains + tests → Streamlit Cloud auto-deploys. "
                "Experiments are tracked with MLflow.")

with tab3:
    st.subheader("Usage monitoring")
    if LOG.exists():
        log = pd.read_csv(LOG)
        st.metric("Total requests", len(log))
        st.metric("Unrecognised-item rate", f"{log['n_unmatched'].sum() / max(log['n_items'].sum(), 1) * 100:.1f}%")
        st.dataframe(log.tail(20))
    else:
        st.info("No requests yet. Use the Optimizer tab first.")
