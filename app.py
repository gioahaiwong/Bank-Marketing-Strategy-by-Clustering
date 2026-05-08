import streamlit as st
import pandas as pd
import plotly.express as px

from model import (
    get_model_results,
    predict_cluster,
    get_cluster_profile,
    get_strategy_recommendation,
    explain_strategy
)

# ========================
# PAGE CONFIG
# ========================
st.set_page_config(
    page_title="Bank Marketing Recommendation System",
    page_icon="🎯",
    layout="wide"
)

# ========================
# HEADER
# ========================
st.title("🎯 Bank Customer Segmentation & Marketing Recommendation System")
st.markdown(
    "Comparative clustering + intelligent strategy scoring for banking marketing decisions."
)

# ========================
# SIDEBAR INPUT
# ========================
st.sidebar.header("📥 Input Customer Data")

education_mapping = {
    "Elementary School": "basic.4y",
    "Middle School": "basic.6y",
    "High School": "high.school",
    "Professional Course": "professional.course",
    "Bachelor Degree": "university.degree",
    "Unknown": "unknown"
}

st.sidebar.subheader("Basic Information")
age = st.sidebar.slider("Age", 18, 95, 30)
job = st.sidebar.selectbox(
    "Job",
    [
        "admin.", "blue-collar", "entrepreneur", "housemaid",
        "management", "retired", "self-employed", "services",
        "student", "technician", "unemployed", "unknown"
    ]
)
marital = st.sidebar.selectbox("Marital", ["single", "married", "divorced", "unknown"])
education_label = st.sidebar.selectbox("Education Level", list(education_mapping.keys()))
education = education_mapping[education_label]

st.sidebar.subheader("Financial Information")
default = st.sidebar.selectbox("Default Credit", ["yes", "no", "unknown"])
housing = st.sidebar.selectbox("Housing Loan", ["yes", "no", "unknown"])
loan = st.sidebar.selectbox("Personal Loan", ["yes", "no", "unknown"])

st.sidebar.subheader("Campaign Information")
contact = st.sidebar.selectbox("Contact Type", ["cellular", "telephone"])
month = st.sidebar.selectbox(
    "Last Contact Month",
    ["jan", "feb", "mar", "apr", "may", "jun", "jul", "aug", "sep", "oct", "nov", "dec"]
)
day_of_week = st.sidebar.selectbox("Last Contact Day", ["mon", "tue", "wed", "thu", "fri"])
duration = st.sidebar.slider("Last Contact Duration", 0, 5000, 200)
campaign = st.sidebar.slider("Campaign Contacts", 1, 50, 2)
pdays = st.sidebar.slider("Days Since Last Contact", 0, 999, 999)
previous = st.sidebar.slider("Previous Contacts", 0, 10, 0)
poutcome = st.sidebar.selectbox("Previous Outcome", ["success", "failure", "nonexistent"])

st.sidebar.subheader("Economic Indicators")
emp_var_rate = st.sidebar.number_input("Employment Variation Rate", -5.0, 2.0, 1.1)
cons_price_idx = st.sidebar.number_input("Consumer Price Index", 90.0, 100.0, 93.0)
cons_conf_idx = st.sidebar.number_input("Consumer Confidence Index", -60.0, 0.0, -40.0)
euribor3m = st.sidebar.number_input("Euribor 3 Month Rate", 0.0, 6.0, 4.8)
nr_employed = st.sidebar.number_input("Number of Employees", 4000.0, 6000.0, 5200.0)

analyze_btn = st.sidebar.button("🔍 Analyze Customer", use_container_width=True)

# ========================
# MODEL COMPARISON
# ========================
st.subheader("📊 Clustering Model Comparison")

model_results = get_model_results()
comparison_df = pd.DataFrame(model_results["comparison"])

kmeans_row = comparison_df[comparison_df["model"] == "K-Means"].iloc[0]
dbscan_row = comparison_df[comparison_df["model"] == "DBSCAN"].iloc[0]

col1, col2, col3, col4 = st.columns(4)

with col1:
    st.metric("K-Means Silhouette", round(kmeans_row["silhouette"], 3))
    st.metric("K-Means DBI", round(kmeans_row["dbi"], 3))

with col2:
    st.metric("DBSCAN Silhouette", round(dbscan_row["silhouette"], 3))
    st.metric("DBSCAN DBI", round(dbscan_row["dbi"], 3))

with col3:
    st.metric("Metric Best Model", model_results["metric_best_model"])
    st.metric("Deployable Model", model_results["final_deployable_model"])

with col4:
    st.metric("Final K", model_results["final_k"])
    st.metric("DBSCAN Noise Ratio", round(dbscan_row["noise_ratio"], 3))

with st.expander("View detailed model comparison"):
    st.dataframe(comparison_df, use_container_width=True)
    st.info(model_results["notes"])

st.divider()

# ========================
# INPUT DATAFRAME
# ========================
input_df = pd.DataFrame([{
    "age": age,
    "job": job,
    "marital": marital,
    "education": education,
    "default": default,
    "housing": housing,
    "loan": loan,
    "contact": contact,
    "month": month,
    "day_of_week": day_of_week,
    "duration": duration,
    "campaign": campaign,
    "pdays": pdays,
    "previous": previous,
    "poutcome": poutcome,
    "emp.var.rate": emp_var_rate,
    "cons.price.idx": cons_price_idx,
    "cons.conf.idx": cons_conf_idx,
    "euribor3m": euribor3m,
    "nr.employed": nr_employed
}])

# ========================
# RESULT SECTION
# ========================
if analyze_btn:
    cluster, used_model = predict_cluster(input_df)
    profile = get_cluster_profile(cluster)
    ranked_strategy = get_strategy_recommendation(cluster)

    st.subheader("📌 Segmentation Result")

    r1, r2, r3 = st.columns(3)

    with r1:
        st.metric("Predicted Cluster", cluster)

    with r2:
        st.metric("Prediction Model", used_model)

    with r3:
        st.metric("Top Strategy", ranked_strategy[0][0] if ranked_strategy else "N/A")

    st.caption(
        "DBSCAN is used for comparative validation, while K-Means is used for real-time customer prediction."
    )

    st.divider()

    # ========================
    # CLUSTER PROFILE
    # ========================
    st.subheader("👥 Cluster Profile Summary")

    if profile is not None:
        p1, p2, p3, p4 = st.columns(4)

        with p1:
            st.metric("Customer Count", int(profile.get("customer_count", 0)))
            st.metric("Avg Age", round(profile.get("age", 0), 2))

        with p2:
            st.metric("Avg Campaign", round(profile.get("campaign", 0), 2))
            st.metric("Avg Duration", round(profile.get("duration", 0), 2))

        with p3:
            st.metric("Avg Previous", round(profile.get("previous", 0), 2))
            st.metric("Avg Pdays", round(profile.get("pdays", 0), 2))

        with p4:
            st.metric("Avg Euribor", round(profile.get("euribor3m", 0), 2))
            st.metric("Avg Employment Rate", round(profile.get("emp.var.rate", 0), 2))

        with st.expander("View full cluster profile"):
            st.dataframe(pd.DataFrame([profile]), use_container_width=True)

    else:
        st.warning("Cluster profile not found.")

    st.divider()

    # ========================
    # STRATEGY RANKING
    # ========================
    st.subheader("🏆 Ranked Marketing Strategy Recommendation")

    if ranked_strategy:
        strategy_df = pd.DataFrame(ranked_strategy, columns=["Strategy", "Score"])
        strategy_df["Score"] = strategy_df["Score"].round(3)

        left, right = st.columns([1, 1.3])

        with left:
            st.dataframe(strategy_df, use_container_width=True)

            top_strategy = ranked_strategy[0][0]
            top_score = ranked_strategy[0][1]

            st.success(f"Top Recommendation: {top_strategy}")
            st.write(explain_strategy(top_strategy))
            st.caption(f"Confidence Score: {round(top_score, 3)}")

        with right:
            fig = px.bar(
                strategy_df,
                x="Strategy",
                y="Score",
                text="Score",
                title="Marketing Strategy Ranking"
            )
            fig.update_layout(
                height=420,
                xaxis_title="Strategy",
                yaxis_title="Score",
                template="plotly_white"
            )
            fig.update_traces(textposition="outside")
            st.plotly_chart(fig, use_container_width=True)

    else:
        st.warning("No strategy recommendation found for this cluster.")

else:
    st.info("Fill the customer data in the sidebar, then click **Analyze Customer**.")