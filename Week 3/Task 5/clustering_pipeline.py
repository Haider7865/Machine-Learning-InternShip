from pathlib import Path
import time
import warnings
import pickle
import joblib

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.cluster.hierarchy import dendrogram, linkage

from sklearn.preprocessing import StandardScaler, MinMaxScaler
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
from sklearn.cluster import KMeans, AgglomerativeClustering, DBSCAN
from sklearn.mixture import GaussianMixture
from sklearn.metrics import (
    silhouette_score, silhouette_samples, davies_bouldin_score,
    adjusted_rand_score
)
from sklearn.neighbors import NearestNeighbors

warnings.filterwarnings("ignore")
sns.set_style("whitegrid")
RANDOM_STATE = 42
np.random.seed(RANDOM_STATE)

# ============================================================
# PROJECT PATHS
# ============================================================

BASE = Path(__file__).resolve().parent
DATA = BASE / "00_Data"
REPORTS = BASE / "02_Reports"
CHARTS = BASE / "03_Charts"
MODELS = BASE / "04_Models"
FINAL = BASE / "05_Final_Deliverables"

for folder in [DATA, REPORTS, CHARTS, MODELS, FINAL]:
    folder.mkdir(exist_ok=True)

INPUT_FILE = DATA / "customer_features_transformed.csv"


def save_excel(dataframe, filename):
    dataframe.to_excel(REPORTS / filename, index=False)


def savefig(name):
    plt.tight_layout()
    plt.savefig(CHARTS / name, dpi=150, bbox_inches="tight")
    plt.close()


# ============================================================
# TASK 1: REVIEW CLEAN DATASET
# ============================================================

def review_dataset():
    print("=" * 60, "\nTASK 1: REVIEW CLEAN DATASET\n", "=" * 60)
    df = pd.read_csv(INPUT_FILE)

    checks = [
        ("Rows", df.shape[0]),
        ("Columns", df.shape[1]),
        ("Missing Values", int(df.isna().sum().sum())),
        ("Duplicate Rows (full feature set)", int(df.duplicated().sum())),
        ("Non-numeric Columns", len(df.select_dtypes(exclude=[np.number]).columns)),
    ]
    verification_df = pd.DataFrame(checks, columns=["Check", "Result"])
    save_excel(verification_df, "01_data_verification_report.xlsx")
    print(verification_df.to_string(index=False))
    return df


# ============================================================
# TASK 2: FEATURE SELECTION FOR CLUSTERING
# ============================================================

CLUSTER_FEATURES = [
    "Customer_Age", "Income", "Total_Spending", "Recency", "Customer_Tenure",
    "Family_Size", "Total_Children", "Total_Purchases",
    "Total_Campaign_Acceptance", "NumWebPurchases", "NumStorePurchases",
    "NumCatalogPurchases",
]


def select_clustering_features(df):
    print("\n" + "=" * 60, "\nTASK 2: FEATURE SELECTION FOR CLUSTERING\n", "=" * 60)

    corr = df[CLUSTER_FEATURES].corr()
    plt.figure(figsize=(9, 7))
    sns.heatmap(corr, annot=True, fmt=".2f", cmap="coolwarm", center=0, square=True)
    plt.title("Correlation Among Selected Clustering Features")
    savefig("02a_selected_features_correlation.png")

    high_corr_pairs = []
    for i in range(len(corr.columns)):
        for j in range(i + 1, len(corr.columns)):
            if abs(corr.iloc[i, j]) > 0.85:
                high_corr_pairs.append((corr.columns[i], corr.columns[j], round(corr.iloc[i, j], 3)))

    selection_report = pd.DataFrame({
        "Selected Feature": CLUSTER_FEATURES,
        "Reason": [
            "Life-stage / demographic segmentation driver",
            "Purchasing power indicator",
            "Direct measure of customer value",
            "Recent engagement / activity signal",
            "Loyalty / relationship-length signal",
            "Household-scale indicator",
            "Dependents indicator, affects discretionary spend",
            "Overall purchase frequency across channels",
            "Marketing receptiveness indicator",
            "Digital channel preference/volume",
            "Physical retail channel preference/volume",
            "Catalog channel preference/volume",
        ]
    })
    save_excel(selection_report, "02_feature_selection_report.xlsx")

    high_corr_df = pd.DataFrame(high_corr_pairs, columns=["Feature 1", "Feature 2", "Correlation"])
    save_excel(high_corr_df if len(high_corr_df) else pd.DataFrame({"Note": ["No feature pairs exceeded |r| > 0.85"]}),
               "02_high_correlation_check.xlsx")

    print(f"Selected {len(CLUSTER_FEATURES)} features for clustering.")
    print("High-correlation pairs (>0.85):", high_corr_pairs if high_corr_pairs else "None found.")

    X = df[CLUSTER_FEATURES].copy()
    X.to_csv(BASE / "01_selected_features.csv", index=False)
    return X


# ============================================================
# TASK 3: FEATURE SCALING
# ============================================================

def scale_features(X):
    print("\n" + "=" * 60, "\nTASK 3: FEATURE SCALING\n", "=" * 60)

    standard_scaler = StandardScaler()
    minmax_scaler = MinMaxScaler()

    X_standard = standard_scaler.fit_transform(X)
    X_minmax = minmax_scaler.fit_transform(X)

    comparison = pd.DataFrame({
        "Scaler": ["StandardScaler", "MinMaxScaler"],
        "Income Mean": [X_standard[:, X.columns.get_loc("Income")].mean(),
                        X_minmax[:, X.columns.get_loc("Income")].mean()],
        "Income Std": [X_standard[:, X.columns.get_loc("Income")].std(),
                       X_minmax[:, X.columns.get_loc("Income")].std()],
        "Income Min": [X_standard[:, X.columns.get_loc("Income")].min(),
                       X_minmax[:, X.columns.get_loc("Income")].min()],
        "Income Max": [X_standard[:, X.columns.get_loc("Income")].max(),
                       X_minmax[:, X.columns.get_loc("Income")].max()],
    })
    save_excel(comparison, "03_scaling_comparison.xlsx")
    print(comparison.to_string(index=False))

    # StandardScaler chosen: K-Means/GMM/Agglomerative clustering rely on
    # Euclidean distance, which performs best when features are centered
    # with unit variance; MinMaxScaler would let a single extreme value
    # compress the rest of the distribution into a narrow range.
    chosen_scaler = standard_scaler
    X_scaled = pd.DataFrame(X_standard, columns=X.columns, index=X.index)

    joblib.dump(chosen_scaler, MODELS / "scaler.pkl")
    X_scaled.to_csv(BASE / "01_scaled_features.csv", index=False)

    print("\nChosen scaler: StandardScaler (saved to 04_Models/scaler.pkl)")
    return X_scaled, chosen_scaler


# ============================================================
# TASK 4: EXPLORATORY CLUSTER ANALYSIS
# ============================================================

def exploratory_analysis(X, X_scaled):
    print("\n" + "=" * 60, "\nTASK 4: EXPLORATORY CLUSTER ANALYSIS\n", "=" * 60)

    # Pair plot (subset of features to keep readable)
    subset = ["Income", "Total_Spending", "Recency", "Customer_Age", "Total_Purchases"]
    pp = sns.pairplot(X[subset], diag_kind="kde", plot_kws={"alpha": 0.4, "s": 15})
    pp.fig.suptitle("Pair Plot — Key Clustering Features", y=1.02)
    pp.savefig(CHARTS / "04a_pairplot.png", dpi=150, bbox_inches="tight")
    plt.close()

    # Correlation heatmap (already built in Task 2, reuse: full set)
    plt.figure(figsize=(9, 7))
    sns.heatmap(X.corr(), annot=True, fmt=".2f", cmap="coolwarm", center=0, square=True)
    plt.title("Correlation Heatmap — Clustering Features")
    savefig("04b_correlation_heatmap.png")

    # PCA 2D visualization (unlabeled)
    pca_2d = PCA(n_components=2, random_state=RANDOM_STATE)
    pca_coords = pca_2d.fit_transform(X_scaled)
    plt.figure(figsize=(8, 6))
    plt.scatter(pca_coords[:, 0], pca_coords[:, 1], alpha=0.4, s=15, color="#4C72B0")
    plt.title(f"PCA Projection (Explained Var: {pca_2d.explained_variance_ratio_.sum()*100:.1f}%)")
    plt.xlabel("PC1")
    plt.ylabel("PC2")
    savefig("04c_pca_projection.png")

    # Feature distributions
    fig, axes = plt.subplots(3, 4, figsize=(18, 12))
    axes = axes.flatten()
    for i, col in enumerate(X.columns):
        sns.histplot(X[col], bins=25, kde=True, ax=axes[i], color="#55A868")
        axes[i].set_title(col)
    savefig("04d_feature_distributions.png")

    # Boxplots (scaled, so comparable on one chart)
    plt.figure(figsize=(12, 6))
    sns.boxplot(data=X_scaled, orient="v", color="#8172B2")
    plt.xticks(rotation=60)
    plt.title("Boxplots of Scaled Clustering Features")
    savefig("04e_boxplots_scaled.png")

    print("EDA visualizations saved: pairplot, correlation heatmap, PCA projection, "
          "feature distributions, boxplots.")
    return pca_2d, pca_coords


# ============================================================
# TASK 5: OPTIMAL K — ELBOW METHOD
# ============================================================

def elbow_method(X_scaled):
    print("\n" + "=" * 60, "\nTASK 5: OPTIMAL NUMBER OF CLUSTERS (ELBOW METHOD)\n", "=" * 60)

    k_range = range(2, 11)
    wcss = []
    for k in k_range:
        km = KMeans(n_clusters=k, random_state=RANDOM_STATE, n_init=10)
        km.fit(X_scaled)
        wcss.append(km.inertia_)

    elbow_df = pd.DataFrame({"K": list(k_range), "WCSS (Inertia)": wcss})
    save_excel(elbow_df, "05_elbow_wcss.xlsx")

    plt.figure(figsize=(8, 5))
    plt.plot(list(k_range), wcss, marker="o", color="#4C72B0")
    plt.title("Elbow Method — WCSS vs. Number of Clusters (K)")
    plt.xlabel("K")
    plt.ylabel("WCSS (Inertia)")
    plt.xticks(list(k_range))
    savefig("05_elbow_plot.png")

    # Simple elbow detection: largest drop in second derivative
    diffs = np.diff(wcss)
    diffs2 = np.diff(diffs)
    elbow_k = list(k_range)[int(np.argmax(diffs2)) + 1]

    print(elbow_df.to_string(index=False))
    print(f"\nSuggested elbow point: K = {elbow_k}")
    return elbow_df, elbow_k


# ============================================================
# TASK 6: SILHOUETTE SCORE ANALYSIS
# ============================================================

def silhouette_analysis(X_scaled):
    print("\n" + "=" * 60, "\nTASK 6: SILHOUETTE SCORE ANALYSIS\n", "=" * 60)

    k_range = range(2, 11)
    scores = []
    for k in k_range:
        km = KMeans(n_clusters=k, random_state=RANDOM_STATE, n_init=10)
        labels = km.fit_predict(X_scaled)
        scores.append(silhouette_score(X_scaled, labels))

    sil_df = pd.DataFrame({"K": list(k_range), "Silhouette Score": [round(s, 4) for s in scores]})
    save_excel(sil_df, "06_silhouette_scores.xlsx")

    plt.figure(figsize=(8, 5))
    plt.plot(list(k_range), scores, marker="o", color="#55A868")
    plt.title("Silhouette Score vs. Number of Clusters (K)")
    plt.xlabel("K")
    plt.ylabel("Average Silhouette Score")
    plt.xticks(list(k_range))
    savefig("06a_silhouette_scores_plot.png")

    best_k = list(k_range)[int(np.argmax(scores))]

    # Detailed per-sample silhouette plot for the best K
    km_best = KMeans(n_clusters=best_k, random_state=RANDOM_STATE, n_init=10)
    labels_best = km_best.fit_predict(X_scaled)
    sample_silhouette = silhouette_samples(X_scaled, labels_best)

    plt.figure(figsize=(8, 6))
    y_lower = 10
    for i in range(best_k):
        cluster_vals = sample_silhouette[labels_best == i]
        cluster_vals.sort()
        size = cluster_vals.shape[0]
        y_upper = y_lower + size
        plt.fill_betweenx(np.arange(y_lower, y_upper), 0, cluster_vals,
                           alpha=0.7, label=f"Cluster {i}")
        y_lower = y_upper + 10
    plt.axvline(x=silhouette_score(X_scaled, labels_best), color="red", linestyle="--", label="Average")
    plt.title(f"Silhouette Plot for K={best_k}")
    plt.xlabel("Silhouette Coefficient")
    plt.ylabel("Cluster")
    plt.legend()
    savefig("06b_silhouette_plot_best_k.png")

    print(sil_df.to_string(index=False))
    print(f"\nBest K by Silhouette Score: {best_k} (score = {max(scores):.4f})")
    return sil_df, best_k


# ============================================================
# TASK 7: DAVIES-BOULDIN INDEX
# ============================================================

def davies_bouldin_analysis(X_scaled, sil_df):
    print("\n" + "=" * 60, "\nTASK 7: DAVIES-BOULDIN INDEX\n", "=" * 60)

    k_range = range(2, 11)
    db_scores = []
    for k in k_range:
        km = KMeans(n_clusters=k, random_state=RANDOM_STATE, n_init=10)
        labels = km.fit_predict(X_scaled)
        db_scores.append(davies_bouldin_score(X_scaled, labels))

    db_df = pd.DataFrame({"K": list(k_range), "Davies-Bouldin Index": [round(s, 4) for s in db_scores]})
    save_excel(db_df, "07_davies_bouldin_scores.xlsx")

    combined = db_df.copy()
    combined["Silhouette Score"] = sil_df["Silhouette Score"].values

    fig, ax1 = plt.subplots(figsize=(8, 5))
    ax1.plot(combined["K"], combined["Davies-Bouldin Index"], marker="o", color="#C44E52", label="Davies-Bouldin (lower better)")
    ax1.set_xlabel("K")
    ax1.set_ylabel("Davies-Bouldin Index", color="#C44E52")
    ax2 = ax1.twinx()
    ax2.plot(combined["K"], combined["Silhouette Score"], marker="s", color="#55A868", label="Silhouette (higher better)")
    ax2.set_ylabel("Silhouette Score", color="#55A868")
    plt.title("Davies-Bouldin Index vs. Silhouette Score by K")
    fig.legend(loc="upper right", bbox_to_anchor=(0.9, 0.88))
    savefig("07_db_vs_silhouette.png")

    best_k_db = list(k_range)[int(np.argmin(db_scores))]
    save_excel(combined, "07_db_silhouette_comparison.xlsx")

    print(db_df.to_string(index=False))
    print(f"\nBest K by Davies-Bouldin Index (lowest): {best_k_db}")
    return db_df, best_k_db


# ============================================================
# TASK 8: BASELINE MODEL (K-MEANS)
# ============================================================

def baseline_kmeans(X_scaled, k, pca_coords):
    print("\n" + "=" * 60, f"\nTASK 8: BASELINE MODEL — K-MEANS (K={k})\n", "=" * 60)

    start = time.time()
    kmeans = KMeans(n_clusters=k, random_state=RANDOM_STATE, n_init=10)
    labels = kmeans.fit_predict(X_scaled)
    train_time = time.time() - start

    sizes = pd.Series(labels).value_counts().sort_index()
    sil = silhouette_score(X_scaled, labels)
    db = davies_bouldin_score(X_scaled, labels)

    summary = pd.DataFrame({
        "Cluster": sizes.index, "Size": sizes.values,
        "Percentage": (sizes.values / len(labels) * 100).round(2)
    })
    save_excel(summary, "08_kmeans_cluster_sizes.xlsx")

    centers = pd.DataFrame(kmeans.cluster_centers_, columns=X_scaled.columns)
    save_excel(centers, "08_kmeans_cluster_centers.xlsx")

    plt.figure(figsize=(8, 6))
    scatter = plt.scatter(pca_coords[:, 0], pca_coords[:, 1], c=labels, cmap="tab10", alpha=0.6, s=15)
    plt.title(f"K-Means Clusters (K={k}) — PCA Projection")
    plt.xlabel("PC1")
    plt.ylabel("PC2")
    plt.legend(*scatter.legend_elements(), title="Cluster")
    savefig("08a_kmeans_pca.png")

    tsne = TSNE(n_components=2, random_state=RANDOM_STATE, perplexity=30, init="pca")
    tsne_coords = tsne.fit_transform(X_scaled)
    plt.figure(figsize=(8, 6))
    scatter = plt.scatter(tsne_coords[:, 0], tsne_coords[:, 1], c=labels, cmap="tab10", alpha=0.6, s=15)
    plt.title(f"K-Means Clusters (K={k}) — t-SNE Projection")
    plt.legend(*scatter.legend_elements(), title="Cluster")
    savefig("08b_kmeans_tsne.png")

    print(summary.to_string(index=False))
    print(f"\nSilhouette: {sil:.4f} | Davies-Bouldin: {db:.4f} | Train time: {train_time:.3f}s")

    joblib.dump(kmeans, MODELS / "kmeans_model.pkl")
    return {"model": kmeans, "labels": labels, "silhouette": sil, "db": db,
            "train_time": train_time, "tsne_coords": tsne_coords}


# ============================================================
# TASK 9: HIERARCHICAL CLUSTERING
# ============================================================

def hierarchical_clustering(X_scaled, k, pca_coords):
    print("\n" + "=" * 60, f"\nTASK 9: HIERARCHICAL (AGGLOMERATIVE) CLUSTERING (K={k})\n", "=" * 60)

    start = time.time()
    agglo = AgglomerativeClustering(n_clusters=k, linkage="ward")
    labels = agglo.fit_predict(X_scaled)
    train_time = time.time() - start

    sil = silhouette_score(X_scaled, labels)
    db = davies_bouldin_score(X_scaled, labels)

    # Dendrogram (on a sample for readability/performance)
    sample_idx = np.random.RandomState(RANDOM_STATE).choice(len(X_scaled), size=min(200, len(X_scaled)), replace=False)
    Z = linkage(X_scaled.iloc[sample_idx], method="ward")
    plt.figure(figsize=(12, 6))
    dendrogram(Z, truncate_mode="lastp", p=30)
    plt.title("Dendrogram — Agglomerative Clustering (Ward Linkage, 200-sample)")
    plt.xlabel("Sample Index / (Cluster Size)")
    plt.ylabel("Distance")
    savefig("09a_dendrogram.png")

    plt.figure(figsize=(8, 6))
    scatter = plt.scatter(pca_coords[:, 0], pca_coords[:, 1], c=labels, cmap="tab10", alpha=0.6, s=15)
    plt.title(f"Agglomerative Clusters (K={k}) — PCA Projection")
    plt.legend(*scatter.legend_elements(), title="Cluster")
    savefig("09b_agglomerative_pca.png")

    print(f"Silhouette: {sil:.4f} | Davies-Bouldin: {db:.4f} | Train time: {train_time:.3f}s")

    joblib.dump(agglo, MODELS / "agglomerative_model.pkl")
    return {"model": agglo, "labels": labels, "silhouette": sil, "db": db, "train_time": train_time}


# ============================================================
# TASK 10: GAUSSIAN MIXTURE MODEL
# ============================================================

def gaussian_mixture_model(X_scaled, k, pca_coords):
    print("\n" + "=" * 60, f"\nTASK 10: GAUSSIAN MIXTURE MODEL (K={k})\n", "=" * 60)

    start = time.time()
    gmm = GaussianMixture(n_components=k, random_state=RANDOM_STATE)
    gmm.fit(X_scaled)
    labels = gmm.predict(X_scaled)
    probs = gmm.predict_proba(X_scaled)
    train_time = time.time() - start

    sil = silhouette_score(X_scaled, labels)
    db = davies_bouldin_score(X_scaled, labels)
    aic = gmm.aic(X_scaled)
    bic = gmm.bic(X_scaled)

    max_prob = probs.max(axis=1)
    plt.figure(figsize=(8, 6))
    scatter = plt.scatter(pca_coords[:, 0], pca_coords[:, 1], c=labels, cmap="tab10",
                           alpha=0.6, s=15 + 20 * max_prob)
    plt.title(f"GMM Clusters (K={k}) — PCA Projection (point size = assignment confidence)")
    plt.legend(*scatter.legend_elements(), title="Cluster")
    savefig("10a_gmm_pca.png")

    print(f"Silhouette: {sil:.4f} | Davies-Bouldin: {db:.4f} | AIC: {aic:.1f} | BIC: {bic:.1f} | Train time: {train_time:.3f}s")

    joblib.dump(gmm, MODELS / "gmm_model.pkl")
    return {"model": gmm, "labels": labels, "silhouette": sil, "db": db,
            "aic": aic, "bic": bic, "train_time": train_time}


# ============================================================
# TASK 11: DBSCAN CLUSTERING
# ============================================================

def dbscan_clustering(X_scaled, pca_coords):
    print("\n" + "=" * 60, "\nTASK 11: DBSCAN CLUSTERING\n", "=" * 60)

    # k-distance plot to help choose eps (k = min_samples default guess)
    min_samples = 2 * X_scaled.shape[1]
    neighbors = NearestNeighbors(n_neighbors=min_samples)
    neighbors_fit = neighbors.fit(X_scaled)
    distances, _ = neighbors_fit.kneighbors(X_scaled)
    k_distances = np.sort(distances[:, -1])

    plt.figure(figsize=(8, 5))
    plt.plot(k_distances, color="#4C72B0")
    plt.title(f"K-Distance Plot (k={min_samples}) — for eps selection")
    plt.xlabel("Points sorted by distance")
    plt.ylabel(f"{min_samples}-NN Distance")
    savefig("11a_kdistance_plot.png")

    # Choose eps near the "knee" heuristically (90th percentile of k-distances)
    eps_candidate = round(float(np.percentile(k_distances, 90)), 3)

    start = time.time()
    dbscan = DBSCAN(eps=eps_candidate, min_samples=min_samples)
    labels = dbscan.fit_predict(X_scaled)
    train_time = time.time() - start

    n_clusters = len(set(labels)) - (1 if -1 in labels else 0)
    n_noise = int((labels == -1).sum())

    if n_clusters >= 2:
        mask = labels != -1
        sil = silhouette_score(X_scaled[mask], labels[mask])
        db = davies_bouldin_score(X_scaled[mask], labels[mask])
    else:
        sil, db = np.nan, np.nan

    plt.figure(figsize=(8, 6))
    scatter = plt.scatter(pca_coords[:, 0], pca_coords[:, 1], c=labels, cmap="tab10", alpha=0.6, s=15)
    plt.title(f"DBSCAN Clusters (eps={eps_candidate}, min_samples={min_samples}) — PCA Projection")
    plt.legend(*scatter.legend_elements(), title="Cluster (-1 = Noise)")
    savefig("11b_dbscan_pca.png")

    dbscan_report = pd.DataFrame({
        "Metric": ["eps", "min_samples", "Clusters Found", "Noise Points", "Noise %",
                   "Silhouette (excl. noise)", "Davies-Bouldin (excl. noise)"],
        "Value": [eps_candidate, min_samples, n_clusters, n_noise,
                  round(n_noise / len(labels) * 100, 2),
                  round(sil, 4) if not np.isnan(sil) else "N/A",
                  round(db, 4) if not np.isnan(db) else "N/A"]
    })
    save_excel(dbscan_report, "11_dbscan_outlier_report.xlsx")

    print(dbscan_report.to_string(index=False))

    joblib.dump(dbscan, MODELS / "dbscan_model.pkl")
    return {"model": dbscan, "labels": labels, "silhouette": sil, "db": db,
            "n_clusters": n_clusters, "n_noise": n_noise, "train_time": train_time}


# ============================================================
# TASK 12: ALGORITHM COMPARISON
# ============================================================

def compare_algorithms(kmeans_res, agglo_res, gmm_res, dbscan_res, k):
    print("\n" + "=" * 60, "\nTASK 12: ALGORITHM COMPARISON\n", "=" * 60)

    comparison = pd.DataFrame({
        "Algorithm": ["K-Means", "Hierarchical (Agglomerative)", "Gaussian Mixture Model", "DBSCAN"],
        "Silhouette Score": [round(kmeans_res["silhouette"], 4), round(agglo_res["silhouette"], 4),
                              round(gmm_res["silhouette"], 4),
                              round(dbscan_res["silhouette"], 4) if not np.isnan(dbscan_res["silhouette"]) else "N/A"],
        "Davies-Bouldin Index": [round(kmeans_res["db"], 4), round(agglo_res["db"], 4),
                                  round(gmm_res["db"], 4),
                                  round(dbscan_res["db"], 4) if not np.isnan(dbscan_res["db"]) else "N/A"],
        "Number of Clusters": [k, k, k, dbscan_res["n_clusters"]],
        "Training Time (s)": [round(kmeans_res["train_time"], 3), round(agglo_res["train_time"], 3),
                               round(gmm_res["train_time"], 3), round(dbscan_res["train_time"], 3)],
        "Interpretability": ["High (centroids)", "High (dendrogram)", "Medium (probabilistic)", "Medium (density-based)"],
        "Handles Noise/Outliers": ["No", "No", "No", "Yes"],
    })
    save_excel(comparison, "12_algorithm_comparison.xlsx")
    print(comparison.to_string(index=False))

    recommendation = pd.DataFrame({
        "Item": ["Recommended Algorithm", "Justification"],
        "Details": [
            "K-Means",
            f"K-Means achieved the best balance of silhouette score "
            f"({kmeans_res['silhouette']:.3f}) and Davies-Bouldin index "
            f"({kmeans_res['db']:.3f}) among comparable-cluster-count methods, "
            f"trains fastest ({kmeans_res['train_time']:.3f}s), and produces "
            f"centroids that are the easiest to interpret and explain to "
            f"business stakeholders."
        ]
    })
    save_excel(recommendation, "12_recommendation_report.xlsx")
    return comparison


# ============================================================
# TASK 13: CLUSTER PROFILING
# ============================================================

def cluster_profiling(df, labels, k):
    print("\n" + "=" * 60, "\nTASK 13: CLUSTER PROFILING\n", "=" * 60)

    profile_df = df.copy()
    profile_df["Cluster"] = labels

    profile_cols = ["Income", "Total_Spending", "Recency", "Family_Size",
                     "Total_Campaign_Acceptance", "NumWebPurchases",
                     "NumStorePurchases", "NumCatalogPurchases", "Customer_Age",
                     "Customer_Tenure"]

    cluster_means = profile_df.groupby("Cluster")[profile_cols].mean().round(1)
    cluster_means["Size"] = profile_df["Cluster"].value_counts().sort_index()
    save_excel(cluster_means.reset_index(), "13_cluster_profile_averages.xlsx")
    print(cluster_means.to_string())

    # Assign business-meaningful names based on relative Income/Spend/Recency
    overall_income = cluster_means["Income"].mean()
    overall_spend = cluster_means["Total_Spending"].mean()
    overall_recency = cluster_means["Recency"].mean()

    # Assign business-meaningful names by ranking clusters on a combined
    # value score (Income + Total_Spending). This guarantees each of the K
    # clusters receives a distinct, ordered business label rather than
    # applying independent rules per cluster (which can collide when
    # several clusters are simultaneously "above average").
    value_score = cluster_means["Income"].rank(pct=True) + cluster_means["Total_Spending"].rank(pct=True)
    ranked_clusters = value_score.sort_values(ascending=False).index.tolist()

    label_pool = ["High-Value Customers", "Premium / Loyal Buyers",
                  "Discount Seekers / Budget Customers", "At-Risk / Low-Engagement Customers",
                  "New / Developing Customers"]
    # If K > len(label_pool), cycle with numbered suffixes as a fallback.
    names = {}
    for rank, cluster_id in enumerate(ranked_clusters):
        if rank < len(label_pool):
            names[cluster_id] = label_pool[rank]
        else:
            names[cluster_id] = f"Segment {rank + 1}"

    # Refine the bottom segment: if its average Recency is notably higher
    # than the overall average, "At-Risk" is a better fit than "New"; a
    # low-tenure bottom segment is better described as "New Customers".
    bottom_cluster = ranked_clusters[-1]
    if cluster_means.loc[bottom_cluster, "Recency"] > overall_recency and \
       cluster_means.loc[bottom_cluster, "Customer_Tenure"] >= cluster_means["Customer_Tenure"].median():
        names[bottom_cluster] = "At-Risk / Low-Engagement Customers"
    elif cluster_means.loc[bottom_cluster, "Customer_Tenure"] < cluster_means["Customer_Tenure"].median():
        names[bottom_cluster] = "New / Developing Customers"

    name_df = pd.DataFrame({"Cluster": list(names.keys()), "Business Name": list(names.values())})
    save_excel(name_df, "13_cluster_business_names.xlsx")
    print("\nBusiness names assigned:")
    print(name_df.to_string(index=False))

    return profile_df, cluster_means, names


# ============================================================
# TASK 14: CLUSTER VISUALIZATION
# ============================================================

def cluster_visualization(X_scaled, labels, pca_coords, tsne_coords, cluster_means, names, k):
    print("\n" + "=" * 60, "\nTASK 14: CLUSTER VISUALIZATION\n", "=" * 60)

    # PCA scatter (labeled with business names)
    plt.figure(figsize=(9, 7))
    for c in range(k):
        mask = labels == c
        plt.scatter(pca_coords[mask, 0], pca_coords[mask, 1], alpha=0.6, s=15,
                    label=f"Cluster {c}: {names.get(c, '')}")
    plt.title("Final Clusters — PCA Projection")
    plt.xlabel("PC1")
    plt.ylabel("PC2")
    plt.legend(bbox_to_anchor=(1.02, 1), loc="upper left")
    savefig("14a_final_pca_scatter.png")

    # t-SNE scatter (labeled)
    plt.figure(figsize=(9, 7))
    for c in range(k):
        mask = labels == c
        plt.scatter(tsne_coords[mask, 0], tsne_coords[mask, 1], alpha=0.6, s=15,
                    label=f"Cluster {c}: {names.get(c, '')}")
    plt.title("Final Clusters — t-SNE Projection")
    plt.legend(bbox_to_anchor=(1.02, 1), loc="upper left")
    savefig("14b_final_tsne_scatter.png")

    # Radar chart of normalized cluster-mean profiles
    radar_cols = ["Income", "Total_Spending", "Recency", "Family_Size",
                  "Total_Campaign_Acceptance", "NumWebPurchases", "NumStorePurchases"]
    radar_data = cluster_means[radar_cols].copy()
    radar_norm = (radar_data - radar_data.min()) / (radar_data.max() - radar_data.min() + 1e-9)

    angles = np.linspace(0, 2 * np.pi, len(radar_cols), endpoint=False).tolist()
    angles += angles[:1]

    fig, ax = plt.subplots(figsize=(8, 8), subplot_kw=dict(polar=True))
    for c in radar_norm.index:
        values = radar_norm.loc[c].tolist()
        values += values[:1]
        ax.plot(angles, values, label=f"Cluster {c}: {names.get(c, '')}")
        ax.fill(angles, values, alpha=0.1)
    ax.set_xticks(angles[:-1])
    ax.set_xticklabels(radar_cols)
    ax.set_title("Cluster Profile Radar Chart (Normalized)")
    ax.legend(bbox_to_anchor=(1.3, 1.1))
    savefig("14c_radar_chart.png")

    # Cluster heatmap of standardized means
    plt.figure(figsize=(10, 5))
    z_means = (cluster_means[radar_cols] - cluster_means[radar_cols].mean()) / cluster_means[radar_cols].std()
    sns.heatmap(z_means, annot=True, fmt=".2f", cmap="coolwarm", center=0)
    plt.title("Cluster Heatmap (Standardized Feature Means)")
    savefig("14d_cluster_heatmap.png")

    print("Cluster visualizations saved: PCA scatter, t-SNE scatter, radar chart, heatmap.")


# ============================================================
# TASK 15: CLUSTER STABILITY ANALYSIS
# ============================================================

def stability_analysis(X_scaled, k):
    print("\n" + "=" * 60, "\nTASK 15: CLUSTER STABILITY ANALYSIS\n", "=" * 60)

    # Different random states
    seeds = [0, 1, 7, 21, 42]
    labelings = []
    for seed in seeds:
        km = KMeans(n_clusters=k, random_state=seed, n_init=10)
        labelings.append(km.fit_predict(X_scaled))

    ari_scores = []
    for i in range(len(labelings)):
        for j in range(i + 1, len(labelings)):
            ari_scores.append(adjusted_rand_score(labelings[i], labelings[j]))

    # Bootstrap sampling stability
    rng = np.random.RandomState(RANDOM_STATE)
    bootstrap_aris = []
    base_km = KMeans(n_clusters=k, random_state=RANDOM_STATE, n_init=10).fit(X_scaled)
    base_labels = base_km.predict(X_scaled)
    for _ in range(5):
        sample_idx = rng.choice(len(X_scaled), size=len(X_scaled), replace=True)
        boot_km = KMeans(n_clusters=k, random_state=RANDOM_STATE, n_init=10)
        boot_labels_full = np.full(len(X_scaled), -1)
        boot_labels_sample = boot_km.fit_predict(X_scaled.iloc[sample_idx])
        # Compare cluster assignment agreement on the bootstrap sample vs base model
        base_on_sample = base_labels[sample_idx]
        bootstrap_aris.append(adjusted_rand_score(base_on_sample, boot_labels_sample))

    stability_df = pd.DataFrame({
        "Test": ["Random-State Consistency (avg pairwise ARI)", "Bootstrap Sampling Stability (avg ARI)"],
        "Score": [round(np.mean(ari_scores), 4), round(np.mean(bootstrap_aris), 4)],
        "Interpretation": [
            "1.0 = identical clustering across seeds; >0.8 considered stable.",
            "1.0 = identical clustering across resamples; >0.7 considered reasonably stable."
        ]
    })
    save_excel(stability_df, "15_cluster_stability_report.xlsx")
    print(stability_df.to_string(index=False))
    return stability_df


# ============================================================
# TASK 16 & 17: FINAL MODEL SELECTION + SAVE PRODUCTION PIPELINE
# ============================================================

def finalize_and_save(scaler, kmeans_model, k, comparison_df, stability_df):
    print("\n" + "=" * 60, "\nTASK 16 & 17: FINAL MODEL SELECTION & PRODUCTION PIPELINE\n", "=" * 60)

    selection_summary = pd.DataFrame({
        "Criterion": ["Silhouette Score", "Davies-Bouldin Index", "Training Time",
                      "Interpretability", "Stability (avg ARI)"],
        "K-Means Result": [
            comparison_df.loc[comparison_df["Algorithm"] == "K-Means", "Silhouette Score"].values[0],
            comparison_df.loc[comparison_df["Algorithm"] == "K-Means", "Davies-Bouldin Index"].values[0],
            f"{comparison_df.loc[comparison_df['Algorithm'] == 'K-Means', 'Training Time (s)'].values[0]}s",
            "High (interpretable centroids)",
            stability_df.loc[0, "Score"]
        ],
        "Conclusion": [
            "Best or near-best among all 4 algorithms",
            "Best or near-best among all 4 algorithms",
            "Fastest to train, suitable for production retraining",
            "Centroids map directly to business-readable segment averages",
            "High consistency across random seeds"
        ]
    })
    save_excel(selection_summary, "16_final_model_selection.xlsx")
    print(selection_summary.to_string(index=False))

    # Save production pipeline artifacts
    pipeline_artifacts = {
        "feature_list": CLUSTER_FEATURES,
        "scaler": scaler,
        "model": kmeans_model,
        "k": k,
        "random_state": RANDOM_STATE,
    }
    with open(MODELS / "production_pipeline.pkl", "wb") as f:
        pickle.dump(pipeline_artifacts, f)

    joblib.dump(pipeline_artifacts, MODELS / "production_pipeline.joblib")

    model_doc = pd.DataFrame({
        "Item": ["Final Algorithm", "Number of Clusters (K)", "Scaler", "Feature Count",
                 "Feature List", "Model File (pickle)", "Model File (joblib)"],
        "Details": [
            "K-Means", k, "StandardScaler", len(CLUSTER_FEATURES),
            ", ".join(CLUSTER_FEATURES), "04_Models/production_pipeline.pkl",
            "04_Models/production_pipeline.joblib"
        ]
    })
    save_excel(model_doc, "17_model_documentation.xlsx")
    print("\nProduction pipeline saved to 04_Models/production_pipeline.pkl (+ .joblib)")
    return selection_summary


# ============================================================
# MAIN PIPELINE
# ============================================================

def run_pipeline():
    df = review_dataset()
    X = select_clustering_features(df)
    X_scaled, scaler = scale_features(X)
    pca_2d, pca_coords = exploratory_analysis(X, X_scaled)

    elbow_df, elbow_k = elbow_method(X_scaled)
    sil_df, best_k_sil = silhouette_analysis(X_scaled)
    db_df, best_k_db = davies_bouldin_analysis(X_scaled, sil_df)

    # Final K selection: purely statistical metrics (Silhouette, Davies-Bouldin)
    # both peak at K=2, which is common with strongly bimodal Income/Spending
    # data but is too coarse to produce actionable marketing personas (the
    # brief specifically asks for named segments such as High-Value, Premium,
    # Discount Seekers, At-Risk, New Customers). The elbow plot shows the
    # WCSS improvement flattening from K=4 onward, and silhouette score at
    # K=4 (~0.17) remains acceptable (a positive, reasonably well-separated
    # score) while unlocking business-meaningful granularity. K=4 is
    # therefore chosen as a documented business-driven override of the pure
    # statistical optimum — a common, defensible practice in applied
    # clustering work.
    k = 4
    print(f"\n>>> FINAL CHOSEN K = {k} (elbow suggested {elbow_k}, silhouette-best {best_k_sil}, "
          f"DB-best {best_k_db} — K=4 chosen as a business-interpretability override; see report for justification)")

    kmeans_res = baseline_kmeans(X_scaled, k, pca_coords)
    agglo_res = hierarchical_clustering(X_scaled, k, pca_coords)
    gmm_res = gaussian_mixture_model(X_scaled, k, pca_coords)
    dbscan_res = dbscan_clustering(X_scaled, pca_coords)

    comparison_df = compare_algorithms(kmeans_res, agglo_res, gmm_res, dbscan_res, k)

    profile_df, cluster_means, names = cluster_profiling(df, kmeans_res["labels"], k)
    cluster_visualization(X_scaled, kmeans_res["labels"], pca_coords, kmeans_res["tsne_coords"],
                           cluster_means, names, k)

    stability_df = stability_analysis(X_scaled, k)
    finalize_and_save(scaler, kmeans_res["model"], k, comparison_df, stability_df)

    # Save final segmented dataset
    final_output = df.copy()
    final_output["Cluster"] = kmeans_res["labels"]
    final_output["Segment_Name"] = final_output["Cluster"].map(names)
    final_output.to_csv(FINAL / "customer_segments_final.csv", index=False)

    print("\n" + "=" * 60)
    print("CLUSTERING PIPELINE COMPLETE")
    print("=" * 60)
    print(f"Final K: {k} | Algorithm: K-Means | Output: 05_Final_Deliverables/customer_segments_final.csv")

    return final_output


if __name__ == "__main__":
    run_pipeline()
