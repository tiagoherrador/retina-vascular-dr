"""    retina-example --csv train.csv --images path/to/images --out assets/pipeline_examples.png

One row per chosen grade: original photo, the retina found and enhanced, vessels (red), lesions (bright cyan, dark blue)."""
import argparse
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from scipy import ndimage as ndi

from .data import GRADES, load_labels
from .features import detect_bright_lesions, detect_dark_lesions
from .preprocess import load_rgb, prepare
from .vessels import FCUNetSegmenter, classical_vessels


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--csv", required=True)
    ap.add_argument("--images", required=True)
    ap.add_argument("--out", default="assets/pipeline_examples.png")
    ap.add_argument("--grades", type=int, nargs="+", default=[0, 2, 4])
    ap.add_argument("--index", type=int, default=1, help="which image of each grade to use")
    ap.add_argument("--segmenter", choices=["fcunet", "classical"], default="fcunet")
    a = ap.parse_args()
    labels = load_labels(a.csv, a.images)
    seg = FCUNetSegmenter() if a.segmenter == "fcunet" else None
    fig, axes = plt.subplots(len(a.grades), 3, figsize=(9.6, 3.2 * len(a.grades)))
    for row, g in zip(np.atleast_2d(axes), a.grades):
        r = labels[labels["diagnosis"] == g].iloc[a.index]
        raw = load_rgb(r.path)
        prep = prepare(raw)
        vessels = seg.segment(prep) if seg else classical_vessels(prep)
        bright, dark = detect_bright_lesions(prep), detect_dark_lesions(prep, vessels)
        ov_v = prep.rgb.copy(); ov_v[vessels] = (255, 0, 0)
        ov_l = prep.rgb.copy(); ov_l[ndi.binary_dilation(bright)] = (0, 255, 255); ov_l[ndi.binary_dilation(dark)] = (0, 60, 255)
        for ax, im, t in zip(row, (raw, ov_v, ov_l), (f"Original — grade {g} ({GRADES[g]})", "Vessels found", "Lesion candidates")):
            ax.imshow(im); ax.set_title(t, fontsize=9); ax.axis("off")
    Path(a.out).parent.mkdir(parents=True, exist_ok=True)
    fig.tight_layout(); fig.savefig(a.out, dpi=130); plt.close(fig)
    print("saved", a.out)


if __name__ == "__main__":
    main()
