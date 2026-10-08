"""APTOS 2019 labels (id_code, diagnosis 0-4) and image lookup."""
from pathlib import Path

import numpy as np
import pandas as pd

EXTS = (".png", ".jpg", ".jpeg")
GRADES = {0: "No DR", 1: "Mild", 2: "Moderate", 3: "Severe", 4: "Proliferative"}


def load_labels(csv, images_dir) -> pd.DataFrame:
    """Rows of the label file whose image exists in `images_dir` (a subset of APTOS is fine). Adds a `path` column."""
    df = pd.read_csv(csv)
    images = {p.stem: p for p in Path(images_dir).iterdir() if p.suffix.lower() in EXTS}
    df["path"] = df["id_code"].astype(str).map(images)
    found = df.dropna(subset=["path"]).copy()
    found["diagnosis"] = found["diagnosis"].astype(int)
    return found.reset_index(drop=True)


def binary_target(grade, task: str) -> np.ndarray:
    """any_dr: grade >= 1 (any sign of DR). referable_dr: grade >= 2 (moderate or worse; the usual referral threshold)."""
    cut = {"any_dr": 1, "referable_dr": 2}[task]
    return (np.asarray(grade) >= cut).astype(int)
