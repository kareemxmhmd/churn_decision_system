import sys
from pathlib import Path
import streamlit as st

ROOT_DIR = Path(__file__).resolve().parent.parent
if str(ROOT_DIR) not in sys.path:
    sys.path.append(str(ROOT_DIR))

from src.inference import load_inference_artifacts, predict_churn

st.set_page_config(
    page_title="Customer Churn Predictor",
    layout="wide"
)


@st.cache_resource
def get_artifacts():
    return load_inference_artifacts()


def main() -> None:
    st.title("Customer Churn Predictor")
    st.caption("Calibrated XGBoost Pipeline with Cost-Optimal Threshold")

    model, threshold = get_artifacts()

    left_col, right_col = st.columns([1.2, 1], gap="large")


    with left_col:
        with st.form("churn_form"):
            st.subheader("Customer Profile")
            
            col1, col2 = st.columns(2)

            with col1:
                tenure = st.number_input("Tenure (Months)", min_value=0.0, max_value=72.0, value=3.0, step=1.0)
                warehouse_to_home = st.number_input("Warehouse to Home (km)", min_value=1.0, max_value=150.0, value=15.0, step=1.0)
                num_devices = st.number_input("Registered Devices", min_value=1, max_value=10, value=4, step=1)
                num_address = st.number_input("Number of Addresses", min_value=1, max_value=25, value=3, step=1)
                satisfaction = st.slider("Satisfaction Score", min_value=1, max_value=5, value=3)

            with col2:
                days_last_order = st.number_input("Days Since Last Order", min_value=0.0, max_value=60.0, value=2.0, step=1.0)
                cashback = st.number_input("Cashback Amount ($)", min_value=0.0, max_value=500.0, value=145.0, step=5.0)
                order_cat = st.selectbox(
                    "Preferred Order Category",
                    options=["Mobile Phone", "Laptop & Accessory", "Fashion", "Grocery", "Others"],
                )
                marital_status = st.selectbox(
                    "Marital Status",
                    options=["Single", "Married", "Divorced"],
                )
                complain = st.selectbox(
                    "Complained in Last Month?",
                    options=[0, 1],
                    format_func=lambda x: "Yes (1)" if x == 1 else "No (0)",
                )

            submitted = st.form_submit_button("Predict Churn", use_container_width=True)


    with right_col:
        st.subheader("Prediction Result")

        if submitted:
            customer_data = {
                "Tenure": tenure,
                "WarehouseToHome": warehouse_to_home,
                "NumberOfDeviceRegistered": num_devices,
                "SatisfactionScore": satisfaction,
                "NumberOfAddress": num_address,
                "DaySinceLastOrder": days_last_order,
                "CashbackAmount": cashback,
                "PreferedOrderCat": order_cat,
                "MaritalStatus": marital_status,
                "Complain": complain,
            }

            result_df = predict_churn(customer_data, model=model, threshold=threshold)
            row = result_df.iloc[0]

            proba = float(row["churn_probability"])
            pred = int(row["churn_prediction"])

            m1, m2, m3 = st.columns(3)
            m1.metric("Churn Probability", f"{proba:.1%}")
            m2.metric("Optimal Threshold", f"{threshold:.1%}")
            m3.metric("Prediction", "High Risk (1)" if pred == 1 else "Low Risk (0)")

            if pred == 1:
                st.error("⚠️ **Action Recommended:** Customer is likely to churn. Send retention incentive ($50 cost vs $500 LTV loss).")
            else:
                st.success("✅ **No Action Needed:** Customer is likely to stay.")
        else:
            st.info("Please enter customer details and click **Predict Churn**.")


if __name__ == "__main__":
    main()