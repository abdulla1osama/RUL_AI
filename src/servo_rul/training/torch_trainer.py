from pathlib import Path
import math
import torch
from torch import nn
from torch.utils.data import DataLoader


RUL_CAP = 130.0


def asymmetric_weighted_mse(
    pred: torch.Tensor,
    target: torch.Tensor,
    over_weight: float = 1.0,
    eol_weight: float = 0.5,
    eol_threshold: float = 40.0 / RUL_CAP,
) -> torch.Tensor:
    residual = pred - target

    weights = torch.ones_like(target)

    # Penalize overprediction more
    weights = weights + over_weight * (residual > 0).float()

    # Penalize near end-of-life errors more
    weights = weights + eol_weight * (target <= eol_threshold).float()

    return (weights * residual.pow(2)).mean()


def get_scheduler(optimizer, warmup_epochs: int, max_epochs: int):
    min_lr_ratio = 0.05

    def lr_lambda(epoch: int) -> float:
        if epoch < warmup_epochs:
            return (epoch + 1) / warmup_epochs

        progress = (epoch - warmup_epochs) / max(1, max_epochs - warmup_epochs)
        cosine = 0.5 * (1.0 + math.cos(math.pi * progress))
        return max(cosine, min_lr_ratio)

    return torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)


def train_torch_model(
    model: nn.Module,
    train_loader: DataLoader,
    val_loader: DataLoader,
    output_dir: str | Path,
    max_epochs: int,
    lr: float,
    weight_decay: float = 1e-4,
    patience: int = 25,
    warmup_epochs: int = 5,
) -> None:
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    device = torch.device("cpu")
    model = model.to(device)

    optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay)
    scheduler = get_scheduler(optimizer, warmup_epochs, max_epochs)

    best_val_loss = float("inf")
    patience_counter = 0

    for epoch in range(1, max_epochs + 1):
        model.train()
        train_loss = 0.0

        for X_batch, y_batch in train_loader:
            X_batch = X_batch.to(device)
            y_batch = y_batch.to(device) / RUL_CAP

            optimizer.zero_grad()

            predictions = model(X_batch) / RUL_CAP
            loss = asymmetric_weighted_mse(predictions, y_batch)

            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()

            train_loss += loss.item()

        train_loss /= len(train_loader)

        model.eval()
        val_loss = 0.0

        with torch.no_grad():
            for X_batch, y_batch in val_loader:
                X_batch = X_batch.to(device)
                y_batch = y_batch.to(device) / RUL_CAP

                predictions = model(X_batch) / RUL_CAP
                loss = asymmetric_weighted_mse(predictions, y_batch)

                val_loss += loss.item()

        val_loss /= len(val_loader)
        scheduler.step()

        train_rmse_cycles = (train_loss ** 0.5) * RUL_CAP
        val_rmse_cycles = (val_loss ** 0.5) * RUL_CAP
        current_lr = scheduler.get_last_lr()[0]

        print(
            f"Epoch {epoch:03d} | "
            f"train_loss_norm={train_loss:.6f} | "
            f"val_loss_norm={val_loss:.6f} | "
            f"train_rmse_cycles≈{train_rmse_cycles:.3f} | "
            f"val_rmse_cycles≈{val_rmse_cycles:.3f} | "
            f"lr={current_lr:.2e}"
        )

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            torch.save(model.state_dict(), output_dir / "best_model.pt")
            patience_counter = 0
        else:
            patience_counter += 1

        if patience_counter >= patience:
            print(f"Early stop at epoch {epoch} | best_val_loss={best_val_loss:.6f}")
            break

    model.load_state_dict(torch.load(output_dir / "best_model.pt", map_location="cpu"))