import argparse
from pathlib import Path

import joblib
import matplotlib.pyplot as plt
import pandas as pd

from servo_rul.features.preprocessing import load_manifest, prepare_supervised_splits
from servo_rul.utils.config import load_config


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    return parser.parse_args()


def _data_root(config: dict) -> Path:
    return Path(config["data"]["root"])


def save_predicted_vs_true(
    config: dict,
    figures_dir: Path,
    pred_path: Path,
    out_name: str,
    title: str,
) -> None:
    """Scatter of predicted vs (capped) true RUL for a held-out prediction file.

    The truth file (``test_labels.csv``) is read by the official evaluator on the
    instructor side; here it is only used to draw the diagnostic figure.
    """
    truth_path = _data_root(config) / "test_labels.csv"

    if not pred_path.exists() or not truth_path.exists():
        print(f"Skipping {out_name}: missing {pred_path} or {truth_path}.")
        return

    rul_cap = config["data"]["rul_cap"]
    pred = pd.read_csv(pred_path)
    truth = pd.read_csv(truth_path)

    merged = pred.merge(truth, on=["unit_id", "cycle"])
    merged["rul_true_capped"] = merged["rul"].clip(upper=rul_cap)

    plt.figure(figsize=(7, 6))
    plt.scatter(merged["rul_true_capped"], merged["rul_pred"], s=8, alpha=0.5)
    plt.plot([0, rul_cap], [0, rul_cap], linestyle="--", color="k")
    plt.xlabel("True capped RUL")
    plt.ylabel("Predicted RUL")
    plt.title(title)
    plt.tight_layout()
    plt.savefig(figures_dir / out_name, dpi=200)
    plt.close()
    print(f"Saved {figures_dir / out_name}")


def save_pca_and_gmm_figures(config: dict, figures_dir: Path) -> None:
    manifest = load_manifest(config["data"]["split"])
    indicator_columns = manifest["indicator_columns"]

    train_df, _, _, _, _ = prepare_supervised_splits(
        data_root=config["data"]["root"],
        split_path=config["data"]["split"],
        rul_cap=config["data"]["rul_cap"],
    )

    cluster_dir = Path(config["output"]["dir"]) / "clustering"

    scaler = joblib.load(cluster_dir / "cluster_scaler.joblib")
    pca = joblib.load(cluster_dir / "pca.joblib")
    gmm = joblib.load(cluster_dir / "gmm.joblib")

    X = train_df[indicator_columns]
    X_scaled = scaler.transform(X)
    X_pca = pca.transform(X_scaled)
    clusters = gmm.predict(X_pca)

    plt.figure(figsize=(7, 6))
    plt.scatter(X_pca[:, 0], X_pca[:, 1], c=train_df["hidden_stage"], s=8, alpha=0.6)
    plt.xlabel("PC1")
    plt.ylabel("PC2")
    plt.title("Train PCA Colored by Hidden Stage")
    plt.tight_layout()
    plt.savefig(figures_dir / "train_pca_hidden_stage.png", dpi=200)
    plt.close()
    print(f"Saved {figures_dir / 'train_pca_hidden_stage.png'}")

    plt.figure(figsize=(7, 6))
    plt.scatter(X_pca[:, 0], X_pca[:, 1], c=clusters, s=8, alpha=0.6)
    plt.xlabel("PC1")
    plt.ylabel("PC2")
    plt.title("Train GMM Clusters in PCA Space")
    plt.tight_layout()
    plt.savefig(figures_dir / "train_gmm_clusters_pca.png", dpi=200)
    plt.close()
    print(f"Saved {figures_dir / 'train_gmm_clusters_pca.png'}")


def main() -> None:
    args = parse_args()
    config = load_config(args.config)

    figures_dir = Path("runs/figures")
    figures_dir.mkdir(parents=True, exist_ok=True)

    # Required predicted-vs-true figure uses the submitted predictions file (M1).
    save_predicted_vs_true(
        config,
        figures_dir,
        pred_path=Path("runs/student_predictions.csv"),
        out_name="val_predicted_vs_true.png",
        title="Predicted vs True RUL (M1 MLP)",
    )

    # Matching figure for the deep temporal model (M2 CNN), when available.
    save_predicted_vs_true(
        config,
        figures_dir,
        pred_path=Path("runs/m2_predictions.csv"),
        out_name="m2_predicted_vs_true.png",
        title="Predicted vs True RUL (M2 1D-CNN)",
    )

    save_pca_and_gmm_figures(config, figures_dir)

    print(f"Saved figures to: {figures_dir}")


if __name__ == "__main__":
    main()
