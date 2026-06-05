import argparse
from pathlib import Path
from servo_rul.models.clustering import fit_gmm_pipeline
from torch.utils.data import DataLoader
from servo_rul.features.preprocessing import prepare_supervised_splits, load_manifest
from servo_rul.features.windowing import create_sliding_windows, prepare_model_input
from servo_rul.models.linear_baseline import build_linear_model
from servo_rul.models.cnn1d import CNN1DRUL
from servo_rul.models.mlp import MLPRUL
from servo_rul.training.datasets import RULWindowDataset
from servo_rul.training.linear_trainer import train_sklearn_model
from servo_rul.training.torch_trainer import train_torch_model
from servo_rul.utils.config import load_config
from servo_rul.utils.seed import set_seed, seed_worker, make_generator


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    config = load_config(args.config)

    seed = config["training"]["seed"]
    deterministic = config["training"]["deterministic"]

    set_seed(seed=seed, deterministic=deterministic)

    model_name = config["model"]["name"]
    window_size = config["model"]["window"]

    train_df, val_df, _, feature_columns, _ = prepare_supervised_splits(
        data_root=config["data"]["root"],
        split_path=config["data"]["split"],
        rul_cap=config["data"]["rul_cap"],
    )

    manifest = load_manifest(config["data"]["split"])
    indicator_columns = manifest["indicator_columns"]

    X_train, y_train = create_sliding_windows(
        df=train_df,
        feature_columns=feature_columns,
        window_size=window_size,
    )

    X_val, y_val = create_sliding_windows(
        df=val_df,
        feature_columns=feature_columns,
        window_size=window_size,
    )

    output_dir = Path(config["output"]["dir"]) / model_name

    if model_name == "linear":
        X_train_2d = prepare_model_input(X_train, model_type="linear")
        model = build_linear_model()
        train_sklearn_model(
            model=model,
            X_train=X_train_2d,
            y_train=y_train,          # linear stays in original scale
            output_dir=output_dir,
        )

    elif model_name in ("mlp", "cnn1d"):


        if model_name == "mlp":
            X_train = prepare_model_input(X_train, model_type="mlp")
            X_val   = prepare_model_input(X_val,   model_type="mlp")
            model = MLPRUL(
                input_size=X_train.shape[1],
                hidden_sizes=config["model"]["hidden_sizes"],
                dropout=config["model"].get("dropout", 0.2),
            )
        else:
            model = CNN1DRUL(
                num_features=X_train.shape[2],
                channels=config["model"]["channels"],
                kernel_size=config["model"].get("kernel_size", 3),
                dropout=config["model"].get("dropout", 0.2),
            )

        train_loader = DataLoader(
            RULWindowDataset(X_train, y_train),   # ← normalized
            batch_size=config["training"]["batch_size"],
            shuffle=True,
            worker_init_fn=seed_worker,
            generator=make_generator(seed),
        )

        val_loader = DataLoader(
            RULWindowDataset(X_val, y_val),       # ← normalized
            batch_size=config["training"]["batch_size"],
            shuffle=False,
            worker_init_fn=seed_worker,
            generator=make_generator(seed),
        )

        train_torch_model(
            model=model,
            train_loader=train_loader,
            val_loader=val_loader,
            output_dir=output_dir,
            max_epochs=config["training"]["max_epochs"],
            lr=config["training"]["lr"],
            weight_decay=config["training"].get("weight_decay", 1e-4),
        )

    else:
        raise ValueError(f"Unknown model name: {model_name}")

    # ── Clustering ───────────────────────────────────────────────────────────
    cluster_dir = Path(config["output"]["dir"]) / "clustering"
    fit_gmm_pipeline(
        train_df=train_df,
        indicator_columns=indicator_columns,
        output_dir=cluster_dir,
        n_clusters=4,
        pca_components=2,
        random_state=seed,
    )

    print(f"Saved clustering pipeline to: {cluster_dir}")
    print(f"Trained model: {model_name}")
    print(f"Output directory: {output_dir}")


if __name__ == "__main__":
    main()