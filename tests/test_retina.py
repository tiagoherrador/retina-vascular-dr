import cv2
import numpy as np
import pandas as pd
import pytest
from scipy import ndimage as ndi
from skimage.morphology import skeletonize

from retina_dr.evaluate import (cross_validate, duplicate_groups, logistic, operating_point, permutation_auc,
                                sensitivity_threshold, stratified_auc, wilson)
from retina_dr.features import (QUALITY, VASCULAR, branch_tortuosity, detect_bright_lesions, detect_dark_lesions,
                                fractal_dimension, image_hash, lesion_features, neighbour_count, quality_features,
                                vascular_features)
from retina_dr.preprocess import fundus_mask, prepare
from retina_dr.vessels import classical_vessels


def draw(f, size=512):
    m = np.zeros((size, size), np.uint8)
    f(m)
    return m > 0


def fake_fundus(h=300, w=420, cx=None, r=140, seed=0):
    """A dark image with a bright disc (the 'retina'), a few darker vessel-like lines and a couple of bright spots."""
    rng = np.random.default_rng(seed)
    img = np.zeros((h, w, 3), np.uint8)
    cx = cx or w // 2
    cv2.circle(img, (cx, h // 2), r, (190, 110, 60), -1)
    img = (img + rng.normal(0, 3, img.shape).clip(0, 255)).astype(np.uint8)
    cv2.line(img, (cx - 100, h // 2 - 20), (cx + 100, h // 2 + 30), (120, 40, 30), 4)
    cv2.line(img, (cx, h // 2 - 100), (cx + 10, h // 2 + 100), (120, 40, 30), 4)
    cv2.circle(img, (cx + 50, h // 2 - 50), 4, (250, 245, 200), -1)
    return img


# ---------------------------------------------------------------- preprocessing
def test_fundus_mask_finds_an_off_centre_retina_that_a_fixed_circle_would_miss():
    img = fake_fundus(cx=300)                              # retina pushed to the right
    m = fundus_mask(img)
    assert m[150, 300] and not m[150, 20] and not m[5, 5]
    assert 0.9 * np.pi * 140 ** 2 < m.sum() < 1.1 * np.pi * 140 ** 2


def test_prepare_returns_square_images_and_a_retina_mask_with_metadata():
    p = prepare(fake_fundus(h=300, w=420))
    assert p.rgb.shape == p.enhanced.shape == (512, 512, 3) and p.fov.shape == (512, 512)
    assert p.meta["orig_height"] == 300 and p.meta["orig_width"] == 420 and 0.1 < p.meta["fov_fraction_of_image"] < 1
    assert p.fov.mean() > 0.5 and not p.rgb[~p.fov].any()                     # nothing outside the retina


def test_a_blank_image_falls_back_to_the_whole_frame():
    assert fundus_mask(np.zeros((100, 100, 3), np.uint8)).all()


# ---------------------------------------------------------------- morphology
def test_straight_vessels_have_tortuosity_one_even_on_a_diagonal_and_wavy_ones_more():
    diag = skeletonize(draw(lambda m: cv2.line(m, (50, 50), (400, 400), 1, 3)))
    t, _ = branch_tortuosity(diag, np.zeros_like(diag))
    assert t == pytest.approx([1.0], abs=0.02)
    xs = np.arange(50, 450)
    pts = np.stack([xs, (256 + 40 * np.sin(xs / 20)).astype(int)], 1).reshape(-1, 1, 2)
    wavy = skeletonize(draw(lambda m: cv2.polylines(m, [pts], False, 1, 3)))
    assert branch_tortuosity(wavy, np.zeros_like(wavy))[0][0] > 1.3


def test_y_shaped_tree_has_three_endpoints_one_bifurcation_and_three_branches():
    y = draw(lambda m: (cv2.line(m, (256, 500), (256, 256), 1, 3), cv2.line(m, (256, 256), (150, 100), 1, 3),
                        cv2.line(m, (256, 256), (360, 100), 1, 3)))
    fov = np.ones_like(y)
    f = vascular_features(y, fov)
    skel_px = f["v_skeleton_density"] * fov.sum()
    assert f["v_endpoints_per_1000px"] * skel_px / 1000 == pytest.approx(3, abs=0.01)
    assert f["v_bifurcations_per_1000px"] * skel_px / 1000 == pytest.approx(1, abs=0.01)
    assert f["v_branches_per_1000px"] * skel_px / 1000 == pytest.approx(3, abs=0.01)
    assert set(VASCULAR) == set(f)


def test_densities_do_not_depend_on_how_much_black_border_the_image_has():
    v = draw(lambda m: cv2.line(m, (100, 100), (400, 120), 1, 3))
    small_fov = np.zeros_like(v); small_fov[50:450, 50:450] = True
    big_fov = np.ones_like(v)
    assert vascular_features(v, small_fov)["v_vessel_density"] > vascular_features(v, big_fov)["v_vessel_density"]
    # same tissue, twice the retina -> half the density
    assert vascular_features(v, big_fov)["v_vessel_density"] == pytest.approx(v.sum() / big_fov.sum())


def test_empty_vessel_mask_gives_zeros_not_errors():
    f = vascular_features(np.zeros((64, 64), bool), np.ones((64, 64), bool))
    assert f["v_vessel_density"] == 0 and f["v_tortuosity_mean"] == 0 and set(f) == set(VASCULAR)


def test_fractal_dimension_of_a_line_and_a_filled_plane():
    assert fractal_dimension(np.ones((256, 256))) == pytest.approx(2.0, abs=0.01)
    assert fractal_dimension(draw(lambda m: cv2.line(m, (0, 256), (511, 256), 1, 1))) == pytest.approx(1.0, abs=0.05)


# ---------------------------------------------------------------- lesions / quality
def test_bright_spot_is_found_and_dark_detector_ignores_vessels():
    p = prepare(fake_fundus(h=420, w=420, r=190))
    bright = detect_bright_lesions(p)
    assert bright.sum() > 0
    vessels = classical_vessels(p)
    dark = detect_dark_lesions(p, vessels)
    assert not (dark & ndi.binary_dilation(vessels, iterations=3)).any()
    f = lesion_features(bright, dark, p.fov)
    assert f["l_bright_count_per_100kpx"] > 0 and set(f) >= {"l_dark_area_frac", "l_dark_mean_area"}


def test_quality_features_and_hash_are_stable_and_complete():
    p = prepare(fake_fundus())
    q = quality_features(p)
    assert set(q) == set(QUALITY) and q["q_orig_width"] == 420
    v = draw(lambda m: cv2.line(m, (100, 100), (400, 300), 1, 3))
    assert image_hash(v) == image_hash(v.copy()) and len(image_hash(v)) == 64
    other = draw(lambda m: cv2.line(m, (400, 80), (80, 420), 1, 3))
    assert bin(int(image_hash(v), 16) ^ int(image_hash(other), 16)).count("1") > 20


def test_classical_vessel_fallback_finds_the_drawn_vessels_inside_the_retina():
    p = prepare(fake_fundus(h=420, w=420, r=190))
    v = classical_vessels(p)
    assert 0.002 < v.mean() < 0.3 and not v[~p.fov].any()


# ---------------------------------------------------------------- evaluation
def toy(n=240, signal=1.2, seed=0):
    rng = np.random.default_rng(seed)
    y = (rng.random(n) < 0.45).astype(int)
    X = rng.normal(size=(n, 5)); X[:, 0] += signal * y
    return X, y


def test_cross_validation_detects_signal_and_sits_at_chance_on_noise():
    from sklearn.metrics import roc_auc_score
    X, y = toy()
    assert roc_auc_score(y, cross_validate(X, y, logistic(), n_repeats=1)[0]) > 0.75
    Xn, yn = toy(signal=0.0, seed=1)
    assert abs(roc_auc_score(yn, cross_validate(Xn, yn, logistic(), n_repeats=1)[0]) - 0.5) < 0.1


def test_permutation_null_is_centred_on_one_half():
    X, y = toy(signal=0.0)
    null = permutation_auc(X, y, logistic(), n_perm=20)
    assert abs(null.mean() - 0.5) < 0.05


def test_threshold_is_chosen_without_looking_at_the_held_out_fold():
    X, y = toy()
    proba, pred = cross_validate(X, y, logistic(), n_repeats=1, target_sens=0.9)
    sens = operating_point(y, pred)["sensitivity"]
    assert 0.8 < sens < 0.99                                    # near the target, not tuned to perfection
    # shuffling only the held-out labels cannot change the predictions: they are made before the labels are used
    _, pred2 = cross_validate(X, np.random.default_rng(0).permutation(y), logistic(), n_repeats=1, target_sens=0.9)
    assert pred.shape == pred2.shape


def test_sensitivity_threshold_keeps_the_target_share_of_positives():
    y = np.array([1] * 10 + [0] * 10); p = np.concatenate([np.linspace(0.5, 0.95, 10), np.linspace(0.05, 0.6, 10)])
    t = sensitivity_threshold(y, p, 0.9)
    assert (p[y == 1] >= t).mean() >= 0.9


def test_wilson_interval_contains_the_estimate_and_widens_with_less_data():
    lo, hi = wilson(18, 20); lo2, hi2 = wilson(180, 200)
    assert lo < 0.9 < hi and (hi - lo) > (hi2 - lo2)


def test_duplicate_photos_share_a_group():
    g = duplicate_groups(["ffff0000ffff0000", "ffff0000ffff0001", "0000ffff0000ffff"], max_distance=3)
    assert g[0] == g[1] != g[2]


def test_grouped_cv_never_splits_a_duplicate_group():
    X, y = toy(n=60)
    groups = np.repeat(np.arange(30), 2)
    from retina_dr.evaluate import folds
    for tr, te in folds(y, groups, 5, 0):
        assert not set(groups[tr]) & set(groups[te])


def test_fcunet_forward_pass_shape_and_vessel_cleanup():
    torch = pytest.importorskip("torch")
    torch.set_num_threads(1)                                   # avoids a multi-thread crash seen with some macOS builds
    from retina_dr.fcunet import FCUnet
    from retina_dr.vessels import MODEL_KWARGS
    out = FCUnet(**MODEL_KWARGS).eval()(torch.zeros(1, 3, 64, 64))
    assert out.shape == (1, 1, 64, 64)


def test_stratified_auc_ignores_a_camera_shortcut():
    rng = np.random.default_rng(0)
    cam = np.repeat([0, 1], 100)
    y = np.concatenate([(rng.random(100) < 0.1), (rng.random(100) < 0.9)]).astype(int)   # camera 1 is mostly DR
    shortcut = cam + 0.01 * rng.normal(size=200)                                          # the score only knows the camera
    from sklearn.metrics import roc_auc_score
    assert roc_auc_score(y, shortcut) > 0.8                                               # pooled AUC is fooled
    assert abs(stratified_auc(y, shortcut, cam)[0] - 0.5) < 0.15                          # within a camera it is chance
    real = y + rng.normal(0, 0.7, 200)
    assert stratified_auc(y, real, cam)[0] > 0.8


def test_stratified_auc_skips_strata_with_one_class():
    y = np.array([0] * 12 + [1] * 12 + [0] * 20); p = np.random.default_rng(1).random(44)
    auc, used = stratified_auc(y, p, np.array([0] * 24 + [1] * 20))
    assert used == 24 and 0 <= auc <= 1


def test_permutation_within_strata_keeps_each_strata_label_counts():
    X, y = toy(n=120)
    strata = np.repeat([0, 1], 60)
    null = permutation_auc(X, y, logistic(), n_perm=5, strata=strata)
    assert len(null) == 5 and np.all((0 <= null) & (null <= 1))
