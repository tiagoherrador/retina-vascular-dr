"""Honest evaluation on small data: repeated stratified cross-validation, thresholds chosen without peeking,
confidence intervals, a label-permutation null and checks against image-quality shortcuts."""
from itertools import combinations

import numpy as np
import pandas as pd
from scipy.stats import beta
from sklearn.base import clone
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import cohen_kappa_score, roc_auc_score
from sklearn.model_selection import StratifiedGroupKFold, StratifiedKFold
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

from .features import LESIONS, QUALITY, VASCULAR

FEATURE_SETS = {
    "image quality only": QUALITY,
    "vessels": VASCULAR,
    "lesions": LESIONS,
    "vessels + lesions": VASCULAR + LESIONS,
    "vessels + lesions + quality": VASCULAR + LESIONS + QUALITY,
}


def logistic():
    return make_pipeline(StandardScaler(), LogisticRegression(C=0.5, class_weight="balanced", max_iter=2000))


def xgboost(seed: int = 0):
    from xgboost import XGBClassifier
    return XGBClassifier(n_estimators=200, max_depth=3, learning_rate=0.05, subsample=0.8, colsample_bytree=0.8,
                         min_child_weight=3, reg_lambda=5.0, eval_metric="logloss", random_state=seed, n_jobs=1)


MODELS = {"logistic regression": logistic, "gradient boosting (XGBoost)": xgboost}


def _fit_proba(model, X, y):
    m = clone(model)
    if hasattr(m, "scale_pos_weight"):                 # XGBoost: balance the classes
        m.set_params(scale_pos_weight=(y == 0).sum() / max((y == 1).sum(), 1))
    m.fit(X, y)
    return m


def sensitivity_threshold(y, p, target: float = 0.90) -> float:
    """Highest threshold that still catches at least `target` of the positives (so the fewest false alarms)."""
    pos = np.sort(p[y == 1])
    k = int(np.floor((1 - target) * len(pos)))         # positives allowed below the threshold
    return float(pos[k]) if len(pos) else 0.5


def folds(y, groups, n_splits, seed, strat=None):
    """Folds stratified on `strat` (default: the label; pass label x camera to keep every fold's mixture alike)."""
    strat = y if strat is None else strat
    if groups is None:
        return StratifiedKFold(n_splits, shuffle=True, random_state=seed).split(np.zeros(len(y)), strat)
    return StratifiedGroupKFold(n_splits, shuffle=True, random_state=seed).split(np.zeros(len(y)), strat, groups)


def cross_validate(X, y, model, groups=None, n_splits=5, n_repeats=3, seed=0, target_sens: float | None = None, strat=None):
    """Out-of-fold probabilities for every repeat (shape n_repeats x n). With `target_sens`, each training fold picks its
    own threshold from an inner cross-validation (never from the held-out fold) and the held-out predictions are
    thresholded with it; the call then also returns the 0/1 predictions."""
    X, y = np.asarray(X, float), np.asarray(y)
    proba = np.zeros((n_repeats, len(y)))
    pred = np.zeros((n_repeats, len(y)), int)
    for r in range(n_repeats):
        for tr, te in folds(y, groups, n_splits, seed + r, strat):
            m = _fit_proba(model, X[tr], y[tr])
            proba[r, te] = m.predict_proba(X[te])[:, 1]
            if target_sens is not None:
                inner = np.zeros(len(tr))
                for itr, ite in StratifiedKFold(3, shuffle=True, random_state=seed + r).split(X[tr], y[tr]):
                    inner[ite] = _fit_proba(model, X[tr][itr], y[tr][itr]).predict_proba(X[tr][ite])[:, 1]
                pred[r, te] = (proba[r, te] >= sensitivity_threshold(y[tr], inner, target_sens)).astype(int)
    return (proba, pred) if target_sens is not None else proba


def wilson(k: int, n: int, level: float = 0.95):
    """Wilson score interval for a proportion."""
    if n == 0:
        return (float("nan"), float("nan"))
    z = abs(__import__("scipy").stats.norm.ppf((1 - level) / 2))
    p = k / n
    centre = (p + z * z / (2 * n)) / (1 + z * z / n)
    half = z * np.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / (1 + z * z / n)
    return float(centre - half), float(centre + half)


def auc_summary(y, proba, n_boot: int = 1000, seed: int = 0) -> dict:
    """Mean AUC over the repeats, and a bootstrap 95 % interval over images using the repeat-averaged probabilities."""
    aucs = [roc_auc_score(y, p) for p in proba]
    mean_p = proba.mean(axis=0)
    rng = np.random.default_rng(seed)
    boots = []
    for _ in range(n_boot):
        i = rng.integers(0, len(y), len(y))
        if 0 < y[i].sum() < len(i):
            boots.append(roc_auc_score(y[i], mean_p[i]))
    return {"auc": float(np.mean(aucs)), "auc_sd_over_repeats": float(np.std(aucs)),
            "auc_ci95_low": float(np.percentile(boots, 2.5)), "auc_ci95_high": float(np.percentile(boots, 97.5))}


def stratified_auc(y, p, strata, min_per_class: int = 5):
    """AUC computed inside each stratum (here: images of the same size, i.e. almost always the same camera) and combined
    with weights proportional to the number of positive-negative pairs. It answers: given two images from the same
    camera, one with DR and one without, how often does the model score the DR image higher? Strata that lack either
    class (fewer than `min_per_class` of each) cannot contribute. Returns (auc, number of images that contributed)."""
    y, p, strata = np.asarray(y), np.asarray(p), np.asarray(strata)
    num = den = used = 0
    for s in np.unique(strata):
        m = strata == s
        npos, nneg = int(y[m].sum()), int((1 - y[m]).sum())
        if npos >= min_per_class and nneg >= min_per_class:
            num += npos * nneg * roc_auc_score(y[m], p[m]); den += npos * nneg; used += int(m.sum())
    return (num / den if den else float("nan")), used


def mixed_strata_mask(y, strata, min_per_class: int = 5) -> np.ndarray:
    """True for images whose stratum (image size) holds at least `min_per_class` healthy and `min_per_class` DR images."""
    y, strata = np.asarray(y), np.asarray(strata)
    mask = np.zeros(len(y), bool)
    for s in np.unique(strata):
        m = strata == s
        if y[m].sum() >= min_per_class and (1 - y[m]).sum() >= min_per_class:
            mask |= m
    return mask


def stratified_auc_summary(y, proba, strata, n_boot: int = 1000, seed: int = 0, min_per_class: int = 5) -> dict:
    y, strata = np.asarray(y), np.asarray(strata)
    aucs = [stratified_auc(y, p, strata, min_per_class)[0] for p in proba]
    mean_p = proba.mean(axis=0)
    rng = np.random.default_rng(seed)
    boots = []
    for _ in range(n_boot):
        i = rng.integers(0, len(y), len(y))
        v = stratified_auc(y[i], mean_p[i], strata[i], min_per_class)[0]
        if not np.isnan(v):
            boots.append(v)
    return {"strat_auc": float(np.mean(aucs)), "strat_auc_ci95_low": float(np.percentile(boots, 2.5)),
            "strat_auc_ci95_high": float(np.percentile(boots, 97.5)),
            "strat_images_used": stratified_auc(y, mean_p, strata, min_per_class)[1]}


def operating_point(y, pred) -> dict:
    """Sensitivity / specificity averaged over repeats, with Wilson intervals from the average counts."""
    y = np.asarray(y)
    tp = np.mean([((p == 1) & (y == 1)).sum() for p in pred]); fn = (y == 1).sum() - tp
    tn = np.mean([((p == 0) & (y == 0)).sum() for p in pred]); fp = (y == 0).sum() - tn
    lo_s, hi_s = wilson(round(tp), int((y == 1).sum())); lo_p, hi_p = wilson(round(tn), int((y == 0).sum()))
    return {"sensitivity": float(tp / (tp + fn)), "sens_ci95_low": lo_s, "sens_ci95_high": hi_s,
            "specificity": float(tn / (tn + fp)), "spec_ci95_low": lo_p, "spec_ci95_high": hi_p}


def permutation_auc(X, y, model, groups=None, n_perm: int = 50, n_splits: int = 5, seed: int = 0, strata=None) -> np.ndarray:
    """AUC when the labels are shuffled (same pipeline): what 'no signal' looks like for this data size.
    With `strata`, labels are shuffled only among images of the same stratum and the within-stratum AUC is returned."""
    rng = np.random.default_rng(seed)
    y = np.asarray(y)
    out = []
    for i in range(n_perm):
        if strata is None:
            yp = rng.permutation(y)
        else:
            yp = y.copy()
            for st in np.unique(strata):
                idx = np.where(np.asarray(strata) == st)[0]
                yp[idx] = rng.permutation(y[idx])
        p = cross_validate(X, yp, model, None, n_splits, 1, seed + i)[0]
        out.append(roc_auc_score(yp, p) if strata is None else stratified_auc(yp, p, strata)[0])
    return np.array(out)


def _hamming(a: int, b: int) -> int:
    return bin(a ^ b).count("1")


def duplicate_groups(hashes, max_distance: int = 10) -> np.ndarray:
    """Group ids so that (near-)identical images share one id (connected components of pairs within `max_distance` bits).
    Used so that two photos of the same eye never end up on both sides of a split."""
    h = [int(x, 16) for x in hashes]
    parent = list(range(len(h)))

    def find(i):
        while parent[i] != i:
            parent[i] = parent[parent[i]]
            i = parent[i]
        return i
    for i, j in combinations(range(len(h)), 2):
        if _hamming(h[i], h[j]) <= max_distance:
            parent[find(i)] = find(j)
    roots = [find(i) for i in range(len(h))]
    return pd.factorize(np.array(roots))[0]


def qwk(y_true, y_pred) -> float:
    return float(cohen_kappa_score(y_true, y_pred, weights="quadratic"))
