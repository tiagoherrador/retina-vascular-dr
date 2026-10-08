"""Fundus-image preprocessing: find the retina, crop to it, resize and enhance contrast.

APTOS images come from different cameras: some are cut at the top and bottom, some have large black borders and some have
none. A fixed centred circle (the usual shortcut) cuts real retina in some images and keeps black corners in others, so the
retina ("field of view", FOV) is located from the image itself. Everything downstream (vessel density, lesion counts) is
computed inside that mask, so the numbers do not depend on how much black border an image happens to have.
"""
from dataclasses import dataclass

import cv2
import numpy as np
from scipy import ndimage as ndi

SIZE = 512


@dataclass
class Prepared:
    rgb: np.ndarray        # SIZE x SIZE x 3 uint8, cropped to the retina, padded to a square, not enhanced
    enhanced: np.ndarray   # same, with CLAHE on the green channel (this is what the vessel network sees)
    fov: np.ndarray        # SIZE x SIZE bool, True inside the retina
    meta: dict             # original size and how much of the original image is retina


def load_rgb(path) -> np.ndarray:
    img = cv2.imread(str(path), cv2.IMREAD_COLOR)
    if img is None:
        raise FileNotFoundError(f"cannot read image: {path}")
    return cv2.cvtColor(img, cv2.COLOR_BGR2RGB)


def fundus_mask(rgb: np.ndarray, threshold: int = 15) -> np.ndarray:
    """Boolean mask of the retina: not-black pixels, largest connected blob, holes filled."""
    gray = cv2.GaussianBlur(rgb.max(axis=2), (5, 5), 0)
    mask = gray > threshold
    mask = ndi.binary_opening(mask, structure=np.ones((5, 5)))
    lab, n = ndi.label(mask)
    if n == 0:
        return np.ones(rgb.shape[:2], bool)
    sizes = ndi.sum(mask, lab, range(1, n + 1))
    mask = ndi.binary_fill_holes(lab == (1 + int(np.argmax(sizes))))
    return mask if mask.mean() > 0.10 else np.ones(rgb.shape[:2], bool)   # nothing sensible found: use the whole image


def _square(a: np.ndarray, value=0) -> np.ndarray:
    h, w = a.shape[:2]
    s = max(h, w)
    out = np.full((s, s) + a.shape[2:], value, a.dtype)
    out[(s - h) // 2:(s - h) // 2 + h, (s - w) // 2:(s - w) // 2 + w] = a
    return out


def clahe_green(rgb: np.ndarray, tile: int = 17, clip: float = 2.0) -> np.ndarray:
    """CLAHE on the green channel only: the preprocessing the FCU-Net weights were trained with."""
    out = rgb.copy()
    out[..., 1] = cv2.createCLAHE(clipLimit=clip, tileGridSize=(tile, tile)).apply(rgb[..., 1])
    return out


def prepare(rgb: np.ndarray, size: int = SIZE) -> Prepared:
    h, w = rgb.shape[:2]
    mask = fundus_mask(rgb)
    ys, xs = np.where(mask)
    y0, y1, x0, x1 = ys.min(), ys.max() + 1, xs.min(), xs.max() + 1
    crop, m = rgb[y0:y1, x0:x1].copy(), mask[y0:y1, x0:x1]
    crop[~m] = 0
    crop, m = _square(crop), _square(m.astype(np.uint8))
    crop = cv2.resize(crop, (size, size), interpolation=cv2.INTER_AREA)
    m = cv2.resize(m, (size, size), interpolation=cv2.INTER_NEAREST).astype(bool)
    crop[~m] = 0                                              # resizing blends the edge; keep outside strictly black
    meta = {"orig_height": h, "orig_width": w, "fov_fraction_of_image": float(mask.mean())}
    return Prepared(rgb=crop, enhanced=clahe_green(crop), fov=m, meta=meta)
