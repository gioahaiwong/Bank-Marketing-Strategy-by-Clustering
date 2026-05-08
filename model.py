import pandas as pd
import numpy as np
import joblib

kmeans_model = joblib.load("kmeans_model.pkl")
dbscan_model = joblib.load("dbscan_model.pkl")
preprocessor = joblib.load("preprocessor.pkl")

model_results = joblib.load("model_results.pkl")
cluster_profiles = joblib.load("cluster_profiles.pkl")
strategy_scores = joblib.load("strategy_scores.pkl")

selected_features = joblib.load("selected_features.pkl")

def get_model_results():
    return model_results

def predict_cluster(input_df):

    # Ensure correct feature order
    input_df = input_df[selected_features]

    # preprocess
    X_processed = preprocessor.transform(input_df)

    # final deployable model
    deployable_model = model_results.get("final_deployable_model", "K-Means")

    # DBSCAN tidak punya predict() bawaan untuk input baru,
    # jadi deployment tetap menggunakan K-Means
    if deployable_model == "K-Means":
        cluster = kmeans_model.predict(X_processed)[0]
        used_model = "K-Means"
    else:
        cluster = kmeans_model.predict(X_processed)[0]
        used_model = "K-Means (deployment fallback)"

    return cluster, used_model

def get_cluster_profile(cluster_id):
    cluster_id = int(cluster_id)

    # Case 1: dictionary with int key
    if isinstance(cluster_profiles, dict):
        if cluster_id in cluster_profiles:
            return cluster_profiles[cluster_id]

        # Case 2: dictionary with string key
        if str(cluster_id) in cluster_profiles:
            return cluster_profiles[str(cluster_id)]

        # Case 3: dictionary with "Cluster 0" key
        cluster_key = f"Cluster {cluster_id}"
        if cluster_key in cluster_profiles:
            return cluster_profiles[cluster_key]

    # Case 4: dataframe
    if isinstance(cluster_profiles, pd.DataFrame):
        if "cluster" in cluster_profiles.columns:
            row = cluster_profiles[cluster_profiles["cluster"] == cluster_id]
            if not row.empty:
                return row.iloc[0].to_dict()

        if cluster_id in cluster_profiles.index:
            return cluster_profiles.loc[cluster_id].to_dict()

    return None


def get_strategy_recommendation(cluster_id):
    cluster_id = int(cluster_id)

    # Case 1: dictionary with int key
    if isinstance(strategy_scores, dict):
        if cluster_id in strategy_scores:
            scores = strategy_scores[cluster_id]

        elif str(cluster_id) in strategy_scores:
            scores = strategy_scores[str(cluster_id)]

        elif f"Cluster {cluster_id}" in strategy_scores:
            scores = strategy_scores[f"Cluster {cluster_id}"]

        else:
            return []

        # if nested dict
        if isinstance(scores, dict):
            return sorted(scores.items(), key=lambda x: x[1], reverse=True)

        # if already list
        if isinstance(scores, list):
            return scores

    # Case 2: dataframe
    if isinstance(strategy_scores, pd.DataFrame):
        df = strategy_scores.copy()

        if "cluster" in df.columns:
            df = df[df["cluster"] == cluster_id]

        if df.empty:
            return []

        # If dataframe is long format: Strategy, Score
        if "Strategy" in df.columns and "Score" in df.columns:
            return list(df[["Strategy", "Score"]].itertuples(index=False, name=None))

        # If dataframe is wide format
        score_cols = [c for c in df.columns if c != "cluster"]
        row = df.iloc[0]
        scores = {col: row[col] for col in score_cols}
        return sorted(scores.items(), key=lambda x: x[1], reverse=True)

    return []

def explain_strategy(strategy_name):

    explanations = {

        "Awareness Strategy":
        "Recommended for customers with low engagement and limited interaction history.",

        "Conversion Strategy":
        "Recommended for customers with moderate engagement and high conversion potential.",

        "Retention Strategy":
        "Recommended for customers who may require stronger long-term engagement.",

        "High-Value Strategy":
        "Recommended for customers with stronger financial and behavioral indicators."
    }

    return explanations.get(
        strategy_name,
        "Recommended based on customer cluster behavior."
    )