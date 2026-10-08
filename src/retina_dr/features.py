"""Hand-crafted features: vessel morphometry (v_), visible lesions (l_) and image quality / context (q_).

Everything is measured inside the retina mask and expressed per unit of retina (densities, per-length counts), so that
images with more or less black border, or different sizes, stay comparable. Image-quality features are kept separate:
they are never meant to be "findings"; they exist so that I can check whether a model learns the camera instead of the eye.
"""
import cv2
import numpy as np
from scipy import ndimage as ndi
from skimage.graph import route_through_array
from skimage.morphology import skeletonize

from .preprocess import Prepared

_NEIGHBOURS = np.ones((3, 3), int)
_NEIGHBOURS[1, 1] = 0
MIN_BRANCH_PX = 15
MIN_CHORD_PX = 8
K_SIGMA, BRIGHT_FLOOR, DARK_FLOOR = 6.0, 25.0, 15.0

VASCULAR = ["v_vessel_density", "v_skeleton_density", "v_mean_width_px", "v_endpoints_per_1000px",
            "v_bifurcations_per_1000px", "v_branches_per_1000px", "v_branch_length_mean_px", "v_fractal_dimension",
            "v_tortuosity_mean", "v_tortuosity_median", "v_tortuosity_p90"]
LESIONS = ["l_bright_area_frac", "l_bright_count_per_100kpx", "l_bright_mean_area",
           "l_dark_area_frac", "l_dark_count_per_100kpx", "l_dark_mean_area"]
QUALITY = ["q_fov_fraction_of_image", "q_orig_height", "q_orig_width", "q_mean_green", "q_contrast_green",
           "q_sharpness", "q_brightness"]


def neighbour_count(skel: np.ndarray) -> np.ndarray:
    return ndi.convolve(skel.astype(int), _NEIGHBOURS, mode="constant")


def fractal_dimension(mask: np.ndarray) -> float:
    """Box-counting dimension: slope of log(#boxes touching the mask) against log(1 / box size)."""
    z = mask > 0
    if not z.any():
        return 0.0
    n = int(2 ** np.ceil(np.log2(max(z.shape))))
    z = np.pad(z, ((0, n - z.shape[0]), (0, n - z.shape[1])))
    sizes = 2 ** np.arange(int(np.log2(n)) - 1, 0, -1)          # n/2 ... 2
    counts = [np.count_nonzero(z.reshape(n // s, s, n // s, s).any(axis=(1, 3))) for s in sizes]
    keep = np.array(counts) > 0
    return float(-np.polyfit(np.log(sizes[keep]), np.log(np.array(counts)[keep]), 1)[0])


def branch_tortuosity(skel: np.ndarray, junction_zone: np.ndarray):
    """Arc length / straight-line distance for every vessel branch (skeleton between junctions).
    The arc length is the geodesic distance along the branch (diagonal steps count sqrt 2), so a straight diagonal
    vessel gets exactly 1. Returns (tortuosities, arc lengths)."""
    branches = skel & ~junction_zone
    lab, n = ndi.label(branches, structure=np.ones((3, 3)))
    tort, arcs = [], []
    for i, sl in enumerate(ndi.find_objects(lab), start=1):
        comp = lab[sl] == i
        if comp.sum() < MIN_BRANCH_PX:
            continue
        ends = np.argwhere(comp & (neighbour_count(comp) == 1))
        if len(ends) < 2:
            continue                                            # a closed loop: no end-to-end distance
        d = np.linalg.norm(ends[:, None, :] - ends[None, :, :], axis=2)
        a, b = np.unravel_index(np.argmax(d), d.shape)
        chord = d[a, b]
        if chord < MIN_CHORD_PX:
            continue
        cost = np.where(comp, 1.0, 1e6)
        _, arc = route_through_array(cost, tuple(ends[a]), tuple(ends[b]), fully_connected=True, geometric=True)
        tort.append(max(arc, chord) / chord)
        arcs.append(arc)
    return np.array(tort), np.array(arcs)


def vascular_features(vessels: np.ndarray, fov: np.ndarray) -> dict:
    vessels = vessels.astype(bool)
    fov_px = max(int(fov.sum()), 1)
    skel = skeletonize(vessels)
    skel_px = int(skel.sum())
    out = {"v_vessel_density": vessels.sum() / fov_px, "v_skeleton_density": skel_px / fov_px,
           "v_mean_width_px": vessels.sum() / skel_px if skel_px else 0.0, "v_fractal_dimension": fractal_dimension(vessels)}
    if skel_px == 0:
        return {**out, **{k: 0.0 for k in VASCULAR if k not in out}}
    nb = neighbour_count(skel)
    ends = skel & (nb == 1)
    junction = skel & (nb >= 3)
    n_junctions = ndi.label(junction, structure=np.ones((3, 3)))[1]
    zone = ndi.binary_dilation(junction, structure=np.ones((3, 3)))
    tort, arcs = branch_tortuosity(skel, zone)
    per = 1000.0 / skel_px
    out.update({"v_endpoints_per_1000px": ends.sum() * per, "v_bifurcations_per_1000px": n_junctions * per,
                "v_branches_per_1000px": len(arcs) * per,
                "v_branch_length_mean_px": float(arcs.mean()) if len(arcs) else 0.0,
                "v_tortuosity_mean": float(np.average(tort, weights=arcs)) if len(arcs) else 1.0,
                "v_tortuosity_median": float(np.median(tort)) if len(arcs) else 1.0,
                "v_tortuosity_p90": float(np.percentile(tort, 90)) if len(arcs) else 1.0})
    return out


def _components(mask: np.ndarray, min_size: int, max_size: int) -> np.ndarray:
    n, lab, stats, _ = cv2.connectedComponentsWithStats(mask.astype(np.uint8), connectivity=8)
    keep = np.zeros(n, bool)
    keep[1:] = (stats[1:, cv2.CC_STAT_AREA] >= min_size) & (stats[1:, cv2.CC_STAT_AREA] <= max_size)
    return keep[lab]


def _robust_threshold(img: np.ndarray, region: np.ndarray, k: float, floor: float) -> float:
    """median + k robust standard deviations of the response inside `region`, but never below `floor` grey levels.
    (Otsu on a top-hat response lands at the noise level, so it "finds" hundreds of lesions in every image.)"""
    vals = img[region].astype(float)
    if vals.size == 0:
        return 255.0
    med = np.median(vals)
    return float(max(med + k * 1.4826 * np.median(np.abs(vals - med)), floor))


def detect_bright_lesions(prep: Prepared, min_size: int = 10, max_size: int = 500) -> np.ndarray:
    """Bright spots (possible exudates): top-hat of the luminance, thresholded well above the retina's own noise level."""
    inner = ndi.binary_erosion(prep.fov, iterations=15)
    lum = cv2.cvtColor(prep.enhanced, cv2.COLOR_RGB2LAB)[..., 0]
    th = cv2.morphologyEx(lum, cv2.MORPH_TOPHAT, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (15, 15)))
    m = (th > _robust_threshold(th, inner, K_SIGMA, BRIGHT_FLOOR)) & inner
    m = cv2.morphologyEx(m.astype(np.uint8), cv2.MORPH_OPEN, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))) > 0
    return _components(m, min_size, max_size)


def detect_dark_lesions(prep: Prepared, vessels: np.ndarray, min_size: int = 5, max_size: int = 300) -> np.ndarray:
    """Small dark spots (possible microaneurysms / haemorrhages): top-hat of the inverted green channel.
    Vessels are dark too, so they are masked out first; otherwise this would mostly count vessel fragments."""
    inner = ndi.binary_erosion(prep.fov, iterations=15)
    near_vessel = ndi.binary_dilation(vessels.astype(bool), iterations=3)
    region = inner & ~near_vessel
    inv = 255 - prep.enhanced[..., 1]
    th = cv2.morphologyEx(inv, cv2.MORPH_TOPHAT, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (9, 9)))
    m = (th > _robust_threshold(th, region, K_SIGMA, DARK_FLOOR)) & region
    m = cv2.morphologyEx(m.astype(np.uint8), cv2.MORPH_OPEN, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))) > 0
    return _components(m, min_size, max_size)


def lesion_features(bright: np.ndarray, dark: np.ndarray, fov: np.ndarray) -> dict:
    area = max(int(ndi.binary_erosion(fov, iterations=15).sum()), 1)
    out = {}
    for name, m in (("bright", bright), ("dark", dark)):
        n, _, stats, _ = cv2.connectedComponentsWithStats(m.astype(np.uint8), connectivity=8)
        areas = stats[1:, cv2.CC_STAT_AREA]
        out[f"l_{name}_area_frac"] = float(m.sum() / area)
        out[f"l_{name}_count_per_100kpx"] = float((n - 1) * 1e5 / area)
        out[f"l_{name}_mean_area"] = float(areas.mean()) if len(areas) else 0.0
    return out


def quality_features(prep: Prepared) -> dict:
    g = prep.rgb[..., 1].astype(np.float32)
    inside = prep.fov
    gi = g[inside]
    lap = cv2.Laplacian(prep.rgb[..., 1], cv2.CV_32F)
    return {"q_fov_fraction_of_image": prep.meta["fov_fraction_of_image"], "q_orig_height": prep.meta["orig_height"],
            "q_orig_width": prep.meta["orig_width"], "q_mean_green": float(gi.mean()), "q_contrast_green": float(gi.std()),
            "q_sharpness": float(lap[inside].var()), "q_brightness": float(prep.rgb.max(axis=2)[inside].mean())}


def image_hash(vessels: np.ndarray) -> str:
    """256-bit hash (hex) of the vessel pattern: the mask averaged on a 16 x 16 grid, thresholded at its median.
    Two photos of the same eye give (almost) the same hash; different eyes differ in about half of the bits."""
    small = cv2.resize(vessels.astype(np.float32), (16, 16), interpolation=cv2.INTER_AREA).flatten()
    bits = small > np.median(small)
    return f"{int(''.join('1' if b else '0' for b in bits), 2):064x}"


def extract_features(prep: Prepared, vessels: np.ndarray) -> dict:
    bright = detect_bright_lesions(prep)
    dark = detect_dark_lesions(prep, vessels)
    return {**vascular_features(vessels, prep.fov), **lesion_features(bright, dark, prep.fov), **quality_features(prep),
            "hash": image_hash(vessels)}
