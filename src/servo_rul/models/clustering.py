from pathlib import Path
from typing import Any

import joblib
import pandas as pd
from sklearn.decomposition import PCA
from sklearn.mixture import GaussianMixture
from sklearn.preprocessing import StandardScaler


def fit_gmm_pipeline(
    train_df: pd.DataFrame,
    indicator_columns: list[str],
    output_dir: str | Path,
    n_clusters: int = 8,
    pca_components: int = 5,
    random_state: int = 42,
) -> dict[str, Any]:
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    X_train = train_df[indicator_columns]

    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X_train)

    pca = PCA(n_components=pca_components, random_state=random_state)
    X_pca = pca.fit_transform(X_scaled)

    gmm = GaussianMixture(
        n_components=n_clusters,
        covariance_type="full",
        n_init=10,
        random_state=random_state,
    )
    gmm.fit(X_pca)

    joblib.dump(scaler, output_dir / "cluster_scaler.joblib")
    joblib.dump(pca, output_dir / "pca.joblib")
    joblib.dump(gmm, output_dir / "gmm.joblib")

    return {
        "scaler": scaler,
        "pca": pca,
        "gmm": gmm,
    }


def predict_gmm_clusters(
    df: pd.DataFrame,
    indicator_columns: list[str],
    model_dir: str | Path,
) -> pd.DataFrame:
    model_dir = Path(model_dir)

    scaler = joblib.load(model_dir / "cluster_scaler.joblib")
    pca = joblib.load(model_dir / "pca.joblib")
    gmm = joblib.load(model_dir / "gmm.joblib")

    df = df.sort_values(["unit_id", "cycle"]).copy()

    X = df[indicator_columns]
    X_scaled = scaler.transform(X)
    X_pca = pca.transform(X_scaled)

    clusters = gmm.predict(X_pca)

    output_df = df[["unit_id", "cycle"]].copy()
    output_df["cluster"] = clusters

    return output_df