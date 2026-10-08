"""    retina-extract --csv train.csv --images path/to/images --out results/features.csv

Reads each image, finds the retina, segments the vessels (FCU-Net by default) and writes one row of features per image.
Vessel masks can be saved with --masks-dir. Resumable: images already present in --out are skipped."""
import argparse
import time
from pathlib import Path

import cv2
import numpy as np
import pandas as pd

from .data import load_labels
from .features import extract_features
from .preprocess import load_rgb, prepare
from .vessels import FCUNetSegmenter, classical_vessels


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--csv", required=True, help="APTOS labels (id_code, diagnosis)")
    ap.add_argument("--images", required=True)
    ap.add_argument("--out", default="results/features.csv")
    ap.add_argument("--masks-dir", default=None, help="also save the vessel masks here (PNG)")
    ap.add_argument("--segmenter", choices=["fcunet", "classical"], default="fcunet")
    ap.add_argument("--weights", choices=["drive", "all"], default="drive")
    ap.add_argument("--tta", action="store_true", help="average 8 flips/rotations (slower, slightly smoother masks)")
    ap.add_argument("--limit", type=int, default=None)
    a = ap.parse_args()

    labels = load_labels(a.csv, a.images)
    if a.limit:
        labels = labels.head(a.limit)
    out = Path(a.out); out.parent.mkdir(parents=True, exist_ok=True)
    done = pd.read_csv(out) if out.exists() else pd.DataFrame()
    seen = set(done["image"]) if len(done) else set()
    seg = FCUNetSegmenter(a.weights, tta=a.tta) if a.segmenter == "fcunet" else None
    if a.masks_dir:
        Path(a.masks_dir).mkdir(parents=True, exist_ok=True)
    rows, t0 = [], time.time()
    todo = labels[~labels["id_code"].isin(seen)]
    print(f"{len(labels)} labelled images found, {len(todo)} to process")
    for i, r in enumerate(todo.itertuples(), 1):
        prep = prepare(load_rgb(r.path))
        vessels = seg.segment(prep) if seg else classical_vessels(prep)
        if a.masks_dir:
            cv2.imwrite(str(Path(a.masks_dir) / f"{r.id_code}_vessels.png"), vessels.astype(np.uint8) * 255)
        rows.append({"image": r.id_code, "grade": int(r.diagnosis), **extract_features(prep, vessels)})
        if i % 25 == 0 or i == len(todo):
            pd.concat([done, pd.DataFrame(rows)]).to_csv(out, index=False)
            print(f"[{i}/{len(todo)}] {time.time() - t0:.0f}s")
    print(f"features saved to {out}")


if __name__ == "__main__":
    main()
