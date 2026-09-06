import streamlit as st
import pandas as pd
import requests

API_URL_SINGLE = "http://127.0.0.1:8000/predict"
API_URL_BATCH = "http://127.0.0.1:8000/predict_batch"

st.set_page_config(
    page_title="Instant Fraud Engine",
    page_icon="⚡",
    layout="wide"
)

st.title("⚡ Real-Time Fraud Assessment")
st.caption("Upload a file or enter data — scoring runs automatically upon receipt.")

tab1, tab2 = st.tabs(["📁 Instant CSV Upload & Score", "🔍 Live Single Inspector"])

# -------------------------------------------------------------------
# TAB 1: INSTANT BATCH UPLOAD (No buttons needed)
# -------------------------------------------------------------------
with tab1:
    uploaded_file = st.file_uploader(
        "Drop transaction CSV here for immediate evaluation", 
        type=["csv"]
    )
    
    if uploaded_file is not None:
        # 1. Immediately read file
        df = pd.read_csv(uploaded_file)
        
        # Remove true target label if present in upload
        feature_df = df.drop(columns=["Class"], errors="ignore")
        
        # 2. Immediately execute batch scoring
        with st.spinner("Analyzing incoming transactions..."):
            try:
                records = feature_df.to_dict(orient="records")
                response = requests.post(API_URL_BATCH, json=records, timeout=60)
                
                if response.status_code == 200:
                    scored_results = response.json()
                    res_df = pd.DataFrame(scored_results)
                    final_df = pd.concat([df.reset_index(drop=True), res_df], axis=1)

                    # Display Quick Summary Metrics
                    total = len(final_df)
                    flagged = (final_df["decision"] == "BLOCK").sum()
                    approved = total - flagged

                    c1, c2, c3 = st.columns(3)
                    c1.metric("Processed", f"{total:,}")
                    c2.metric("Approved", f"{approved:,}")
                    c3.metric("Blocked / Flagged", f"{flagged:,}", delta=f"{(flagged/total)*100:.1f}% Risk", delta_color="inverse")

                    # Highlight Red for Blocked rows
                    st.subheader("Audited Stream")
                    st.dataframe(
                        final_df.style.map(
                            lambda val: "background-color: #ffcccc; color: #900C3F; font-weight: bold;" if val == "BLOCK" else "",
                            subset=["decision"]
                        ),
                        use_container_width=True
                    )

                    # Download immediately available
                    csv_export = final_df.to_csv(index=False).encode("utf-8")
                    st.download_button(
                        label="Download Scored File",
                        data=csv_export,
                        file_name="auto_scored_transactions.csv",
                        mime="text/csv"
                    )
                else:
                    st.error(f"Backend rejected payload: {response.text}")

            except requests.exceptions.ConnectionError:
                st.error("Cannot reach FastAPI service. Ensure Uvicorn is active on http://127.0.0.1:8000.")

# -------------------------------------------------------------------
# TAB 2: INSTANT LIVE INSPECTION (Live score updates as you type)
# -------------------------------------------------------------------
with tab2:
    st.subheader("Interactive Transaction Profiler")
    
    col1, col2 = st.columns([1, 2])
    with col1:
        preset = st.radio("Simulation Profile", ["Legitimate Activity", "Known Fraud Pattern"])
        amount = st.number_input("Transaction Amount ($)", min_value=0.01, value=149.62 if preset == "Known Fraud Pattern" else 25.00, step=5.0)
        time_elapsed = st.number_input("Seconds Since Start", min_value=0.0, value=406.0 if preset == "Known Fraud Pattern" else 100.0, step=10.0)

    # Feature defaults
    if preset == "Known Fraud Pattern":
        pca_defaults = {
            "V1": -2.31, "V2": 1.95, "V3": -1.60, "V4": 3.99, "V5": -0.52,
            "V6": -1.42, "V7": -2.53, "V8": 1.39, "V9": -2.77, "V10": -2.77,
            "V11": 3.20, "V12": -2.89, "V13": -0.59, "V14": -4.28, "V15": 0.38,
            "V16": -1.14, "V17": -2.83, "V18": -0.01, "V19": 0.41, "V20": 0.12,
            "V21": 0.51, "V22": -0.03, "V23": -0.46, "V24": 0.32, "V25": 0.04,
            "V26": 0.17, "V27": 0.26, "V28": -0.14
        }
    else:
        pca_defaults = {f"V{i}": 0.01 for i in range(1, 29)}

    with col2:
        with st.expander("Feature Vector (V1 - V28)", expanded=False):
            v_inputs = {}
            sub_cols = st.columns(4)
            for i in range(1, 29):
                key = f"V{i}"
                with sub_cols[(i - 1) % 4]:
                    v_inputs[key] = st.number_input(key, value=float(pca_defaults[key]))

    # Automatic prediction call on value update
    single_payload = {
        "Time": float(time_elapsed),
        "Amount": float(amount),
        **v_inputs
    }
    
    try:
        res = requests.post(API_URL_SINGLE, json=single_payload, timeout=5)
        if res.status_code == 200:
            data = res.json()
            prob = data["fraud_probability"]
            decision = data["decision"]
            risk = data["risk_level"]

            st.divider()
            m1, m2, m3 = st.columns(3)
            m1.metric("Fraud Likelihood", f"{prob * 100:.2f}%")
            m2.metric("Automated Decision", decision)
            m3.metric("Risk Level", risk)

            if decision == "BLOCK":
                st.error("🚨 **INTERVENTION REQUIRED**: High likelihood of unauthorized activity.")
            else:
                st.success("✅ **CLEARED**: Standard transaction profile.")
    except requests.exceptions.ConnectionError:
        st.warning("Start the backend server in Terminal 1 to activate live scoring.")