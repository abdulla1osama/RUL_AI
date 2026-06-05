from pathlib import Path
from typing import Any

import joblib
import numpy as np


def train_sklearn_model(model: Any,X_train: np.ndarray,y_train: np.ndarray,output_dir: str | Path,) -> None:
    
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    model.fit(X_train, y_train)

    joblib.dump(model, output_dir / "model.joblib")


def load_sklearn_model(model_path: str | Path) -> Any:
    return joblib.load(model_path)