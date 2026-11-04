import sys
from pathlib import Path
import pandas as pd
import streamlit as st

ROOT_DIR = Path(__file__).resolve().parent.parent
if str(ROOT_DIR) not in sys.path:
    sys.path.append(str(ROOT_DIR))

from src.inference import load_inference_artifacts, predict_churn

st.set_page_config(
    page_title="Customer Churn Decision System",
    layout="wide"
)


@st.cache_resource
def get_artifacts():
    return load_inference_artifacts()


def main() -> None:
    st.title("Customer Churn Decision System")
    st.caption("Calibrated XGBoost Pipeline with Cost-Optimal Threshold")

    model, threshold = get_artifacts()

    tab_single, tab_batch = st.tabs(["👤 Single Customer Prediction", "📁 Batch Prediction (CSV)"])

    with tab_single:
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
                st.info("Please enter customer details on the left and click **Predict Churn**.")

    with tab_batch:
        st.subheader("Batch Prediction via CSV Upload")
        st.write("Upload a CSV file containing customer data to predict churn risk for multiple customers simultaneously.")

        sample_template = pd.DataFrame({
            "Tenure": [3.0, 15.0, 1.0],
            "WarehouseToHome": [15.0, 8.0, 25.0],
            "NumberOfDeviceRegistered": [4, 3, 5],
            "SatisfactionScore": [3, 4, 1],
            "NumberOfAddress": [3, 2, 6],
            "DaySinceLastOrder": [2.0, 12.0, 0.0],
            "CashbackAmount": [145.0, 210.0, 110.0],
            "PreferedOrderCat": ["Mobile Phone", "Fashion", "Laptop & Accessory"],
            "MaritalStatus": ["Single", "Married", "Single"],
            "Complain": [0, 0, 1],
        })

        st.download_button(
            label="Download Sample CSV Template",
            data=sample_template.to_csv(index=False),
            file_name="sample_customers.csv",
            mime="text/csv",
            help="Download this template to see the required column names."
        )

        uploaded_file = st.file_uploader("Choose a CSV file", type=["csv"])

        if uploaded_file is not None:
            try:
                batch_df = pd.read_csv(uploaded_file)
                st.write(f"Loaded **{len(batch_df):,}** customers. Data Preview:")
                st.dataframe(batch_df.head(5), use_container_width=True)

                if st.button("Run Batch Prediction", type="primary", use_container_width=True):
                    with st.spinner("Processing batch predictions..."):
                        preds_df = predict_churn(batch_df, model=model, threshold=threshold)

                        enriched_df = batch_df.copy()
                        enriched_df["churn_probability"] = preds_df["churn_probability"]
                        enriched_df["churn_prediction"] = preds_df["churn_prediction"]
                        enriched_df["risk_category"] = preds_df["churn_prediction"].map({1: "High Risk", 0: "Low Risk"})
                        enriched_df["recommended_action"] = preds_df["churn_prediction"].map({
                            1: "Send retention incentive ($50)",
                            0: "No action needed"
                        })

                        total_count = len(enriched_df)
                        churn_count = int((preds_df["churn_prediction"] == 1).sum())
                        churn_rate = churn_count / total_count if total_count > 0 else 0
                        retention_budget = churn_count * 50

                        st.success("Batch Prediction Complete!")
                        st.subheader("Business Impact Overview")
                        b1, b2, b3, b4 = st.columns(4)
                        b1.metric("Total Customers Evaluated", f"{total_count:,}")
                        b2.metric("High Risk Churners", f"{churn_count:,}")
                        b3.metric("Predicted Churn Rate", f"{churn_rate:.1%}")
                        b4.metric("Retention Budget Needed", f"${retention_budget:,}")

                        st.subheader("Prediction Results Table")
                        st.dataframe(enriched_df, use_container_width=True)

                        csv_data = enriched_df.to_csv(index=False).encode("utf-8")
                        st.download_button(
                            label="Download Full Predictions CSV",
                            data=csv_data,
                            file_name="churn_batch_predictions_results.csv",
                            mime="text/csv",
                            use_container_width=True,
                        )
            except Exception as exc:
                st.error(f"Error during batch processing: {exc}")


if __name__ == "__main__":
    main()