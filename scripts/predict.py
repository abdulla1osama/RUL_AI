import argparse
from pathlib import Path

import pandas as pd
import torch

from servo_rul.models.clustering import predict_gmm_clusters
from servo_rul.features.preprocessing import prepare_supervised_splits, load_manifest
from servo_rul.features.windowing import create_sliding_windows, prepare_model_input
from servo_rul.models.cnn1d import CNN1DRUL
from servo_rul.models.mlp import MLPRUL
from servo_rul.training.linear_trainer import load_sklearn_model
from servo_rul.utils.config import load_config


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    parser.add_argument("--split", required=True)
    parser.add_argument("--out", required=True)
    parser.add_argument("--clusters-out", default=None)
    return parser.parse_args()


def predict_torch_model(model: torch.nn.Module, X_test) -> torch.Tensor:
    device = torch.device("cpu")
    model = model.to(device)
    model.eval()

    X_tensor = torch.tensor(X_test, dtype=torch.float32).to(device)

    with torch.no_grad():
        predictions = model(X_tensor)

    return predictions.cpu()


def make_prediction_index(test_df: pd.DataFrame, window_size: int) -> pd.DataFrame:
    return (
        test_df.sort_values(["unit_id", "cycle"])[["unit_id", "cycle"]]
        .reset_index(drop=True)
    )


def main() -> None:
    args = parse_args()
    config = load_config(args.config)

    model_name = config["model"]["name"]
    window_size = config["model"]["window"]
    rul_cap = float(config["data"]["rul_cap"])

    _, _, test_df, feature_columns, _ = prepare_supervised_splits(
        data_root=config["data"]["root"],
        split_path=args.split,
        rul_cap=config["data"]["rul_cap"],
    )

    X_test, _ = create_sliding_windows(
        df=test_df,
        feature_columns=feature_columns,
        window_size=window_size,
    )

    model_dir = Path(config["output"]["dir"]) / model_name

    if model_name == "linear":
        X_test_model = prepare_model_input(X_test, model_type="linear")
        model = load_sklearn_model(model_dir / "model.joblib")
        raw_predictions = model.predict(X_test_model)
        # linear model predicts in original scale — just clip
        predictions = raw_predictions.clip(0, rul_cap)

    elif model_name == "mlp":
        X_test_model = prepare_model_input(X_test, model_type="mlp")

        model = MLPRUL(
            input_size=X_test_model.shape[1],
            hidden_sizes=config["model"]["hidden_sizes"],
            dropout=config["model"].get("dropout", 0.2),
        )
        model.load_state_dict(
            torch.load(model_dir / "best_model.pt", map_location="cpu", weights_only=True)
        )

        raw_predictions = predict_torch_model(model, X_test_model).numpy()
        # mlp trained on normalized targets — denormalize then clip
        predictions = raw_predictions.clip(0, rul_cap)

    elif model_name == "cnn1d":
        model = CNN1DRUL(
            num_features=X_test.shape[2],
            channels=config["model"]["channels"],
            kernel_size=config["model"].get("kernel_size", 3),
            dropout=config["model"].get("dropout", 0.2),
        )
        model.load_state_dict(
            torch.load(model_dir / "best_model.pt", map_location="cpu", weights_only=True)
        )

        raw_predictions = predict_torch_model(model, X_test).numpy()
        # cnn1d trained on normalized targets — denormalize then clip
        predictions = raw_predictions.clip(0, rul_cap)

    else:
        raise ValueError(f"Unknown model name: {model_name}")

    output_df = make_prediction_index(test_df, window_size)

    if len(output_df) != len(predictions):
        raise ValueError(
            f"Prediction length mismatch: {len(predictions)} predictions "
            f"for {len(output_df)} rows"
        )

    output_df["rul_pred"] = predictions

    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    output_df.to_csv(out_path, index=False)

    print(f"Saved predictions to: {out_path}")
    print(f"Model: {model_name}")
    print(f"Prediction rows: {len(output_df)}")

    if args.clusters_out is not None:
        manifest = load_manifest(args.split)
        indicator_columns = manifest["indicator_columns"]

        cluster_dir = Path(config["output"]["dir"]) / "clustering"

        clusters_df = predict_gmm_clusters(
            df=test_df,
            indicator_columns=indicator_columns,
            model_dir=cluster_dir,
        )

        clusters_out_path = Path(args.clusters_out)
        clusters_out_path.parent.mkdir(parents=True, exist_ok=True)
        clusters_df.to_csv(clusters_out_path, index=False)

        print(f"Saved clusters to: {clusters_out_path}")
        print(f"Cluster rows: {len(clusters_df)}")


if __name__ == "__main__":
    main()