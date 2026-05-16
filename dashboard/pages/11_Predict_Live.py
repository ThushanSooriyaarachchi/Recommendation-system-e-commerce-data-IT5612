"""Page 11 - Live scoring against trained classifier and regressor."""

from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import pandas as pd
import streamlit as st

from src import config
from src.models import load as load_model
from dashboard.components.bootstrap import bootstrap

st.set_page_config(page_title="Predict Live", page_icon="🎯", layout="wide")
df_full, _, _ = bootstrap()
st.title("Live Prediction")


@st.cache_resource(show_spinner="Loading models…")
def load_models():
    clf = load_model(config.CLF_MODEL_PATH) if config.CLF_MODEL_PATH.exists() else None
    reg = load_model(config.REG_MODEL_PATH) if config.REG_MODEL_PATH.exists() else None
    return clf, reg


clf, reg = load_models()
if clf is None and reg is None:
    st.error("Models not trained. Run `python run_pipeline.py` first.")
    st.stop()


def _opts(col: str) -> list[str]:
    return sorted(df_full[col].dropna().astype(str).unique())


with st.form("predict_form"):
    c1, c2, c3 = st.columns(3)
    with c1:
        gender = st.selectbox("Gender", _opts("gender"))
        age = st.slider("Age", 18, 90, 35)
        segment = st.selectbox("Customer segment", _opts("customer_segment"))
        country = st.selectbox("Country", _opts("country"))
    with c2:
        category = st.selectbox("Category", _opts("category"))
        sub_category = st.selectbox("Sub-category", _opts("sub_category"))
        brand = st.selectbox("Brand", _opts("brand"))
        unit_price = st.number_input("Unit price (USD)", 1.0, 5000.0, 75.0, step=5.0)
        quantity = st.number_input("Quantity", 1, 50, 2)
        discount = st.slider("Discount %", 0, 75, 10)
    with c3:
        payment_method = st.selectbox("Payment method", _opts("payment_method"))
        shipping_method = st.selectbox("Shipping method", _opts("shipping_method"))
        delivery_days = st.slider("Delivery days", 1, 30, 5)
        device_type = st.selectbox("Device", _opts("device_type"))
        traffic_source = st.selectbox("Traffic source", _opts("traffic_source"))
        campaign_source = st.selectbox("Campaign source", _opts("campaign_source"))
        order_priority = st.selectbox("Order priority", _opts("order_priority"))
        fraud_risk = st.slider("Fraud risk score", 0, 100, 30)
        loyalty = st.slider("Loyalty score", 0, 100, 50)

    submitted = st.form_submit_button("Score", width="stretch")


if submitted:
    discount_amount = unit_price * quantity * discount / 100
    total = unit_price * quantity - discount_amount
    record = {
        "gender": gender,
        "age": age,
        "customer_segment": segment,
        "country": country,
        "category": category,
        "sub_category": sub_category,
        "brand": brand,
        "payment_method": payment_method,
        "shipping_method": shipping_method,
        "device_type": device_type,
        "traffic_source": traffic_source,
        "campaign_source": campaign_source,
        "order_priority": order_priority,
        "customer_loyalty_score": loyalty,
        "total_orders_by_customer": 10,
        "customer_tenure_days": 365,
        "product_rating_avg": 4.0,
        "product_reviews_count": 1500,
        "stock_quantity": 200,
        "unit_price_usd": unit_price,
        "quantity": quantity,
        "discount_percent": discount,
        "discount_amount_usd": discount_amount,
        "total_price_usd": total,
        "cost_usd": total * 0.6,
        "tax_usd": total * 0.1,
        "shipping_cost_usd": 8.0,
        "delivery_days": delivery_days,
        "session_duration_minutes": 25.0,
        "pages_visited": 8,
        "fraud_risk_score": fraud_risk,
        "profit_margin_percent": 35.0,
        "order_hour": 14,
        "order_dow": 2,
    }
    X = pd.DataFrame([record])

    cols = st.columns(2)
    if clf is not None:
        try:
            X_clf = X.reindex(columns=clf.num_cols + clf.cat_cols)
            proba = float(clf.pipeline.predict_proba(X_clf)[0, 1])
            cols[0].metric("P(returned order)", f"{proba*100:,.1f}%")
            with cols[0]:
                st.progress(min(max(proba, 0), 1))
                if clf.report.roc_auc is not None:
                    st.caption(
                        f"Test set ROC-AUC: {clf.report.roc_auc:.3f} | "
                        f"F1: {clf.report.f1:.3f}"
                    )
                else:
                    st.caption(f"Test set F1: {clf.report.f1:.3f}")
        except Exception as exc:
            cols[0].error(f"Classifier failed: {exc}")

    if reg is not None:
        try:
            X_reg = X.reindex(columns=reg.num_cols + reg.cat_cols)
            pred = float(reg.pipeline.predict(X_reg)[0])
            cols[1].metric("Predicted profit (USD)", f"${pred:,.2f}")
            with cols[1]:
                st.caption(
                    f"Test set R²: {reg.report.r2:.3f} | "
                    f"RMSE: ${reg.report.rmse:,.2f}"
                )
        except Exception as exc:
            cols[1].error(f"Regressor failed: {exc}")
