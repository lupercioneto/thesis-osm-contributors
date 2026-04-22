import os
os.environ["LOKY_MAX_CPU_COUNT"] = "4"  # Set to your number of cores

from pathlib import Path
import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.cluster import KMeans
from sklearn.mixture import GaussianMixture
from sklearn.metrics import silhouette_score, davies_bouldin_score, calinski_harabasz_score
from sklearn.utils import resample
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import datetime
import time
import os
from kneed import KneeLocator


# ----------------------------- CONFIG -------------------------------- #


# Construct path relative to this file's location
PROJECT_ROOT = Path(__file__).parent.parent.parent
print("Base dir:", PROJECT_ROOT)

DATA_PATH = (
    PROJECT_ROOT 
    / "results" 
    / "00_preprocessing" 
    / "user_summary" 
    / "cat.parquet"
).as_posix()

PLOTS_DIR = PROJECT_ROOT / "results" / "02_categorization" / "plots" 
PLOTS_DIR.mkdir(parents=True, exist_ok=True)

RESULTS_DIR = PROJECT_ROOT / "results" / "02_categorization" / "results"
RESULTS_DIR.mkdir(parents=True, exist_ok=True)


# ----------------------------- CLASS AND FUNCTIONS ------------------- #

class OSMClusteringPipeline:
    def __init__(self, parquet_path, output_dir="results", plot_dir="plots"):
        self.parquet_path = parquet_path
        self.output_dir = output_dir
        self.plot_dir = plot_dir
        self.current_time = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")

        os.makedirs(self.output_dir, exist_ok=True)
        os.makedirs(self.plot_dir, exist_ok=True)

        # Container for Data
        self.df_filtered_full = None
        self.df_final = None
        self.X_pca = None
        self.labels_kmeans = None
        self.labels_gmm = None

    # ---------- Helper Functions ----------
    @staticmethod
    def evaluate_clustering(X, labels, sample_size=None):
        if sample_size and X.shape[0] > sample_size:
            rng = np.random.default_rng(42)
            idx = rng.choice(X.shape[0], sample_size, replace=False)
            X_sample = X[idx]
            labels_sample = labels[idx]
        else:
            X_sample = X
            labels_sample = labels
        return {
            'silhouette': float(silhouette_score(X_sample, labels_sample)),
            'davies_bouldin': float(davies_bouldin_score(X_sample, labels_sample)),
            'calinski_harabasz': float(calinski_harabasz_score(X_sample, labels_sample))
        }

    @staticmethod
    def find_optimal_n(data, n_range, threshold=0.01):
        """
        Choose the first n where improvement <= threshold.
        """
        data = np.array(data)
        improvements = (data[:-1] - data[1:]) / np.abs(data[:-1])

        for i, imp in enumerate(improvements):
            print(f"n={n_range[i+1]}, improvement={imp:.4f}")
            if imp <= threshold:
                return n_range[i]  # take the cluster count before it drops below threshold
        return n_range[-1]  # fallback: the last n if never below threshold


    # ---------- Pipeline Steps ----------
    def load_and_filter(self):
        df = pd.read_parquet(self.parquet_path)
        self.df_filtered_full = df[(df["total_edits"] > 10) & (df["left_early"] == False)].copy()

        drop_features = [
            'social_facility_ratio', 'active_week_ratio', 
            'body_of_water_ratio', 'financial_service_ratio', 'wash_facility_ratio', 'place_ratio',
            'waterway_ratio', 
            'user_id', 'first_edit', 'full_last_edit',
            'first_edit_year', 'first_edit_month', 'top_country', 'top_feature_type_name', 'left_early'
        ]

        self.df_filtered_full.drop(columns=drop_features, errors="ignore", inplace=True)
        print("Filtered Data shape:", self.df_filtered_full.shape)

    def transform_features(self):
        df = self.df_filtered_full.copy()
        # Fill missing values
        df["days_to_50"] = df["days_to_50"].fillna(df["active_duration"] + 1)
        df["burstiness_score"] = df["burstiness_score"].fillna(0)
        df["comment_length_ratio"] = df["comment_length_ratio"].fillna(0)
        df["top_feature_ratio"] = df["top_feature_ratio"].fillna(0)
        df["days_to_100"] = df["days_to_100"].fillna(df["active_duration"] + 1)
        df["edits_per_span_day"] = df["edits_per_span_day"].fillna(0)
        df["changesets_per_edit_day"] = df["changesets_per_edit_day"].fillna(0)

        # Apply scaling
        scaled_cols = []
        for col in df.columns:
            if not np.issubdtype(df[col].dtype, np.number):
                continue
            vals = df[col].dropna()
            if vals.empty:
                continue
            min_val, max_val = vals.min(), vals.max()
            skew = vals.skew()
            if min_val >= 0 and max_val <= 1.5:
                transform = "StandardScaler"
            elif skew > 2:
                transform = "log1p + StandardScaler"
            else:
                transform = "StandardScaler"

            if transform == "log1p + StandardScaler":
                vals = np.log1p(df[col])
            else:
                vals = df[col]
            new_col = f"{col}_scaled"
            df[new_col] = StandardScaler().fit_transform(vals.values.reshape(-1, 1))
            scaled_cols.append(new_col)

        self.df_final = df[scaled_cols]
        print("Transformed features shape:", self.df_final.shape)

    def run_pca(self, var_threshold=0.9):
        n_input_features = self.df_final.shape[1]

        pca = PCA(n_components=var_threshold)
        self.X_pca = pca.fit_transform(self.df_final)

        explained = np.cumsum(pca.explained_variance_ratio_)
        n_selected = pca.n_components_

        plt.figure(figsize=(8,5))
        plt.plot(range(1, len(explained)+1), explained, marker='o')
        plt.axhline(var_threshold, color='r', linestyle='--',
                    label=f"{int(var_threshold*100)}% threshold")
        plt.axvline(n_selected, color='g', linestyle='--',
                    label=f"{n_selected} components")
        plt.scatter(n_selected, explained[n_selected - 1], s=60)

        plt.xlabel("Number of principal components")
        plt.ylabel("Cumulative explained variance")
        plt.title("Scree Plot / Variance explained")
        plt.legend()
        plt.tight_layout()
        plt.savefig(f"{self.plot_dir}/pca_scree_plot_{self.current_time}.png")
        plt.close()

        loadings = pd.DataFrame(
            pca.components_.T,
            columns=[f"PC{i+1}" for i in range(pca.n_components_)],
            index=self.df_final.columns
        )
        loadings.to_csv(f"{self.output_dir}/pca_loadings_{self.current_time}.csv")

        print(f"PCA input features: {n_input_features}")
        print(f"Selected components for {int(var_threshold*100)}% explained variance: {n_selected}")
        print("PCA finished, X_pca shape:", self.X_pca.shape)
        reduction_pct = 100 * (1 - n_selected / n_input_features)
        print(f"Dimensionality reduced from {n_input_features} to {n_selected} features ({reduction_pct:.1f}% reduction).")


    def run_kmeans(self, candidate_k=range(2, 16), fixed_n=None, sample_size=10000, selection_method="elbow"):
        distortions = []
        silhouette_results = {}
        results = []
        k_values = list(candidate_k)

        for k in k_values:
            print(f"Testing KMeans for k={k} ...")
            kmeans = KMeans(n_clusters=k, random_state=42, n_init=20)
            labels = kmeans.fit_predict(self.X_pca)
            distortions.append(kmeans.inertia_)

            # Silhouette (subsample if too large)
            if len(self.X_pca) > sample_size:
                X_sample, y_sample = resample(
                    self.X_pca,
                    labels,
                    n_samples=sample_size,
                    random_state=42
                )
                sil = silhouette_score(X_sample, y_sample)
            else:
                sil = silhouette_score(self.X_pca, labels)
            silhouette_results[k] = sil

            # Other metrics
            metrics = self.evaluate_clustering(self.X_pca, labels, sample_size=sample_size)
            metrics["k"] = k
            metrics["inertia"] = kmeans.inertia_
            results.append(metrics)

        ch_scores = [res["calinski_harabasz"] for res in results]
        db_scores = [res["davies_bouldin"] for res in results]

        # choose k
        kl = KneeLocator(
            k_values,
            distortions,
            curve="convex",
            direction="decreasing"
        )
        best_k_elbow = kl.elbow
        best_k_silhouette = max(silhouette_results, key=silhouette_results.get)
        best_k_davies_bouldin = min(zip(k_values, db_scores), key=lambda x: x[1])[0]
        best_k_calinski_harabasz = max(zip(k_values, ch_scores), key=lambda x: x[1])[0]

        print("Best k (Elbow):", best_k_elbow)
        print("Best k (Silhouette):", best_k_silhouette)
        print("Best k (Davies-Bouldin):", best_k_davies_bouldin)
        print("Best k (Calinski-Harabasz):", best_k_calinski_harabasz)

        # Save results
        pd.DataFrame(results).set_index("k").to_csv(
            f"{self.output_dir}/metrics_kmeans_{self.current_time}.csv"
        )

        # Silhouette plot
        plt.figure(figsize=(6, 4))
        plt.plot(k_values, list(silhouette_results.values()), marker="o")
        plt.xlabel("Number of clusters")
        plt.ylabel("Silhouette Score")
        plt.title("Silhouette Scores for KMeans")
        plt.grid(True)
        plt.tight_layout()
        plt.savefig(f"{self.plot_dir}/silhouette_kmeans_{self.current_time}.png")
        plt.close()

        # Elbow plot
        plt.figure(figsize=(6, 4))
        plt.plot(k_values, distortions, marker="o", label="Inertia")
        if best_k_elbow is not None:
            elbow_idx = k_values.index(best_k_elbow)
            plt.axvline(best_k_elbow, color="red", linestyle="--", label=f"Kneedle elbow = {best_k_elbow}")
            plt.scatter(best_k_elbow, distortions[elbow_idx], color="red", zorder=5)
        if selection_method == "fixed" and fixed_n is not None and fixed_n in k_values:
            fixed_idx = k_values.index(fixed_n)
            plt.axvline(fixed_n, color="green", linestyle="--", label=f"Selected k = {fixed_n}")
            plt.scatter(fixed_n, distortions[fixed_idx], color="green", zorder=5)
        plt.xlabel("Number of clusters (k)")
        plt.ylabel("Distortion (Inertia)")
        plt.title("Elbow Method")
        plt.grid(True)
        plt.legend()
        plt.tight_layout()
        plt.savefig(f"{self.plot_dir}/elbow_kmeans_{self.current_time}.png")
        plt.close()

        # Davies-Bouldin plot
        plt.figure(figsize=(6, 4))
        plt.plot(k_values, db_scores, marker="o")
        plt.xlabel("Number of clusters (k)")
        plt.ylabel("Davies-Bouldin Score")
        plt.title("Davies-Bouldin Scores for KMeans")
        plt.grid(True)
        plt.tight_layout()
        plt.savefig(f"{self.plot_dir}/davies_bouldin_kmeans_{self.current_time}.png")
        plt.close()

        # Calinski-Harabasz plot
        plt.figure(figsize=(6, 4))
        plt.plot(k_values, ch_scores, marker="o")
        plt.xlabel("Number of clusters (k)")
        plt.ylabel("Calinski-Harabasz Score")
        plt.title("Calinski-Harabasz Scores for KMeans")
        plt.grid(True)
        plt.tight_layout()
        plt.savefig(f"{self.plot_dir}/calinski_harabasz_kmeans_{self.current_time}.png")
        plt.close()

        # choose final k
        if selection_method == "elbow":
            best_k = best_k_elbow if best_k_elbow is not None else best_k_silhouette
        elif selection_method == "silhouette":
            best_k = best_k_silhouette
        elif selection_method == "davies_bouldin":
            best_k = best_k_davies_bouldin
        elif selection_method == "calinski_harabasz":
            best_k = best_k_calinski_harabasz
        elif selection_method == "fixed":
            if fixed_n is None:
                raise ValueError("fixed_n must be provided when selection_method='fixed'")
            best_k = fixed_n
        else:
            raise ValueError(
                "selection_method must be one of: "
                "'elbow', 'silhouette', 'davies_bouldin', 'calinski_harabasz', 'fixed'"
            )

        print(f"Final selected k: {best_k}")

        # Final clustering with chosen k
        kmeans_final = KMeans(n_clusters=best_k, random_state=42, n_init=20)
        self.labels_kmeans = kmeans_final.fit_predict(self.X_pca)

        df_cluster = self.df_final.copy()
        df_cluster["cluster_kmeans"] = self.labels_kmeans
        cluster_means = df_cluster.groupby("cluster_kmeans").mean()
        cluster_means.to_csv(f"{self.output_dir}/cluster_means_all_features{self.current_time}.csv")

        top_features = cluster_means.var(axis=0).sort_values(ascending=False).head(20).index
        plt.figure(figsize=(20, 12))
        sns.heatmap(cluster_means[top_features], cmap="coolwarm", annot=True, fmt=".2f")
        plt.title(f"Cluster Profiles (Top 20 Features) K={best_k}")
        plt.tight_layout()
        plt.savefig(f"{self.plot_dir}/cluster_profiles_kmeans_{self.current_time}.png")
        plt.close()

        df_cluster.to_parquet(f"{self.output_dir}/kmeans_clusters_{self.current_time}.parquet")
        print("KMeans cluster profiles saved.")

        return best_k, cluster_means

    def run_gmm(self, n_range=range(2, 16), fixed_n=None, criterion="bic"):
        """
        Do the GMM clustering.
        - If fixed_n is set -> use that value.
        - Else -> select best n via AIC or BIC.
        """
        print("\n--- Starting GMM clustering ---")
        gmm_start_total = time.time()

        if fixed_n is None:
            bics, aics = [], []
            print(f"Testing candidate numbers of components: {list(n_range)}")

            for n in n_range:
                start_n = time.time()
                print(f"[GMM] Fitting model with n_components={n} ...")

                gmm = GaussianMixture(
                    n_components=n,
                    covariance_type='full',
                    random_state=42
                )
                gmm.fit(self.X_pca)

                bic = gmm.bic(self.X_pca)
                aic = gmm.aic(self.X_pca)

                bics.append(bic)
                aics.append(aic)

                runtime_n = time.time() - start_n
                print(
                    f"[GMM] Done n={n} | "
                    f"BIC={bic:.2f} | AIC={aic:.2f} | "
                    f"runtime={runtime_n:.2f}s"
                )

            print("[GMM] Candidate model fitting finished.")

            plt.figure(figsize=(8,5))
            plt.plot(n_range, bics, marker='o', label="BIC")
            plt.plot(n_range, aics, marker='s', label="AIC")

            # Select best n based on criterion
            if criterion.lower() == "bic":
                best_n = self.find_optimal_n(bics, list(n_range), threshold=0.01)
                print(f"[GMM] Best GMM n selected by BIC rule: {best_n}")
            else:
                best_n = self.find_optimal_n(aics, list(n_range), threshold=0.01)
                print(f"[GMM] Best GMM n selected by AIC rule: {best_n}")
            # mark selected number of components
            if best_n in n_range:
                idx = list(n_range).index(best_n)
                plt.axvline(best_n, color="red", linestyle="--", label=f"Selected k = {best_n}")
                if criterion.lower() == "bic":
                    plt.scatter(best_n, bics[idx], color="red", zorder=5)
                else:
                    plt.scatter(best_n, aics[idx], color="red", zorder=5)

            plt.xlabel("Number of components")
            plt.ylabel("Information Criterion")
            plt.title("Model selection with BIC/AIC for GMM")
            plt.legend()
            plt.grid(True)
            plt.tight_layout()
            plt.savefig(f"{self.plot_dir}/gmm_bic_aic_{self.current_time}.png")
            plt.close()

            pd.DataFrame({
                "n_components": list(n_range),
                "BIC": bics,
                "AIC": aics
            }).to_csv(f"{self.output_dir}/gmm_bic_aic_{self.current_time}.csv", index=False)


        else:
            best_n = fixed_n
            print(f"[GMM] Using manually fixed n_components={best_n}")

        print(f"[GMM] Fitting final model with n_components={best_n} ...")
        final_start = time.time()

        gmm_final = GaussianMixture(
            n_components=best_n,
            covariance_type='full',
            random_state=42
        )
        self.labels_gmm = gmm_final.fit_predict(self.X_pca)

        print(f"[GMM] Final model finished in {time.time() - final_start:.2f}s")

        # Analyze and save cluster profiles
        df_cluster = self.df_final.copy()
        df_cluster["cluster_gmm"] = self.labels_gmm

        cluster_means = (
            df_cluster.groupby("cluster_gmm")
            .mean(numeric_only=True)
            .reset_index()
        )

        cluster_means.to_csv(
            f"{self.output_dir}/gmm_cluster_means_all_features_{self.current_time}.csv",
            index=False
        )

        # Number of features for Heatmap can be adjusted
        n_feat = 25
        top_features = cluster_means.var(axis=0).sort_values(ascending=False).head(n_feat).index

        # PLOT CLUSTER PROFILES
        plt.figure(figsize=(20,14))
        sns.heatmap(cluster_means[top_features], cmap="coolwarm", annot=True, fmt=".2f")
        plt.title(f"Cluster Profiles (Top {n_feat} Features) with GMM n={best_n} ({criterion.upper()})")
        plt.tight_layout()
        plt.savefig(f"{self.plot_dir}/cluster_profiles_gmm_{self.current_time}.png")
        plt.close()

        # Cluster size
        cluster_counts = df_cluster["cluster_gmm"].value_counts().sort_index()
        cluster_share = cluster_counts / cluster_counts.sum() * 100

        summary_df = pd.DataFrame({
            "Cluster": cluster_counts.index,
            "Size": cluster_counts.values,
            "Share (%)": cluster_share.values.round(2)
        })
        summary_df.to_csv(f"{self.output_dir}/gmm_cluster_size_{self.current_time}.csv", index=False)
        
        # Save final clustered data
        df_cluster.to_parquet(f"{self.output_dir}/gmm_clusters_{self.current_time}.parquet")

        print(f"--- GMM clustering completed in {time.time() - gmm_start_total:.2f}s ---\n")



# ---------- RUN SCRIPT ----------
if __name__ == "__main__":
    start = time.time()

    pipeline = OSMClusteringPipeline(
        parquet_path=DATA_PATH,
        output_dir=RESULTS_DIR,
        plot_dir=PLOTS_DIR
    )

    pipeline.load_and_filter()
    pipeline.transform_features()
    pipeline.run_pca(var_threshold=0.9)
    print("PCA completed: ", round(time.time() - start, 2), "seconds")
    pipeline.run_kmeans()
    print("KMeans completed: ", round(time.time() - start, 2), "seconds")
    pipeline.run_gmm()
    print("GMM completed: ", round(time.time() - start, 2), "seconds")

    print("Total runtime:", round(time.time() - start, 2), "seconds")
