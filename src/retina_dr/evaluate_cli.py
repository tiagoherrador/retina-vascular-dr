"""    retina-evaluate --features results/features.csv --out results

Repeated stratified 5-fold cross-validation on all labelled images (duplicate photos never straddle a split):
- healthy vs any DR, and non-referable vs referable DR (grade >= 2), for several feature sets and two models
- AUC both pooled and WITHIN image size (a proxy for the camera): in this subset the image size alone nearly gives the label away
- sensitivity-first operating point (threshold chosen inside the training folds only)
- label-permutation null, an image-quality-only baseline, 5-grade results, SHAP for the final model"""
import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score, confusion_matrix, f1_score, roc_auc_score
from sklearn.model_selection import StratifiedKFold

from . import plots
from .data import GRADES, binary_target
from .evaluate import (FEATURE_SETS, stratified_auc, MODELS, auc_summary, cross_validate, duplicate_groups, mixed_strata_mask, operating_point, permutation_auc, qwk,
                       stratified_auc_summary)

TASKS = {"any_dr": "Healthy vs any DR (grade ≥ 1)", "referable_dr": "Non-referable vs referable DR (grade ≥ 2)"}
FINAL_SET, TARGET_SENS = "vessels + lesions", 0.90


def five_class(df, cols, repeats, seed, mixed):
    from xgboost import XGBClassifier
    X, y = df[cols].values.astype(float), df["grade"].values
    rows, cm_total = [], np.zeros((5, 5))
    makers = {"XGBoost, 5 classes": lambda: XGBClassifier(n_estimators=200, max_depth=3, learning_rate=0.05, subsample=0.8,
                                                         colsample_bytree=0.8, min_child_weight=2, reg_lambda=5.0, n_jobs=1, random_state=0)}
    try:
        import mord
        from sklearn.pipeline import make_pipeline
        from sklearn.preprocessing import StandardScaler
        makers["Ordinal logistic (mord)"] = lambda: make_pipeline(StandardScaler(), mord.LogisticAT(alpha=1.0))
    except ImportError:
        pass
    for name, make in makers.items():
        acc, f1m, k, kmix = [], [], [], []
        for r in range(repeats):
            pred = np.zeros(len(y), int)
            for tr, te in StratifiedKFold(5, shuffle=True, random_state=seed + r).split(X, y):
                pred[te] = make().fit(X[tr], y[tr]).predict(X[te])
            acc.append(accuracy_score(y, pred)); f1m.append(f1_score(y, pred, average="macro")); k.append(qwk(y, pred)); kmix.append(qwk(y[mixed], pred[mixed]))
            if r == 0:
                cm = confusion_matrix(y, pred, labels=range(5))
                cm_total = cm if "XGB" in name else cm_total
        rows.append({"model": name, "accuracy": np.mean(acc), "macro_f1": np.mean(f1m), "qwk": np.mean(k), "qwk_sd": np.std(k), "qwk_mixed_sizes_only": np.mean(kmix)})
    base = np.zeros(len(y), int) + np.bincount(y).argmax()
    rows.append({"model": "Always predict the most common grade", "accuracy": accuracy_score(y, base),
                 "macro_f1": f1_score(y, base, average="macro"), "qwk": qwk(y, base), "qwk_sd": 0.0, "qwk_mixed_sizes_only": qwk(y[mixed], base[mixed])})
    return pd.DataFrame(rows), cm_total


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--features", default="results/features.csv")
    ap.add_argument("--out", default="results")
    ap.add_argument("--repeats", type=int, default=3)
    ap.add_argument("--perms", type=int, default=50)
    ap.add_argument("--no-shap", action="store_true")
    a = ap.parse_args()
    out = Path(a.out); out.mkdir(parents=True, exist_ok=True)

    df = pd.read_csv(a.features).dropna(subset=["grade"]).reset_index(drop=True)
    groups = duplicate_groups(df["hash"]) if "hash" in df else None
    size = df["q_orig_width"].astype(int).astype(str) + "x" + df["q_orig_height"].astype(int).astype(str)   # camera proxy
    size_code = pd.factorize(size)[0]
    n_dup = int(len(df) - len(set(groups))) if groups is not None else 0
    print(f"{len(df)} images; grades {df['grade'].value_counts().sort_index().to_dict()}; images sharing a group with another: {n_dup}")
    summary = {"n_images": len(df), "grade_counts": {int(k): int(v) for k, v in df["grade"].value_counts().sort_index().items()},
               "images_in_duplicate_groups": n_dup, "repeats": a.repeats, "distinct_image_sizes": int(size.nunique())}
    ct = pd.crosstab(size, df["grade"] > 0)
    summary["size_vs_any_dr"] = {k: {"healthy": int(r[False]), "dr": int(r[True])} for k, r in ct.iterrows()}
    ct.rename(columns={False: "healthy", True: "any_dr"}).assign(total=ct.sum(axis=1)).sort_values("total", ascending=False).to_csv(out / "image_size_vs_label.csv")

    # --- AUC for every task x feature set x model
    rows, store = [], {}
    for task in TASKS:
        y = binary_target(df["grade"], task)
        strat = y + 2 * size_code                      # folds keep the same label x camera mixture
        for fs, cols in FEATURE_SETS.items():
            X = df[cols].values.astype(float)
            for mname, make in MODELS.items():
                proba = cross_validate(X, y, make(), groups, 5, a.repeats, seed=0, strat=strat)
                rows.append({"task": task, "features": fs, "model": mname, **auc_summary(y, proba),
                             **stratified_auc_summary(y, proba, size)})
                store[(task, fs, mname)] = (y, proba.mean(axis=0))
                print(f"{task:13s} {fs:28s} {mname:28s} AUC {rows[-1]['auc']:.3f}  within-size AUC {rows[-1]['strat_auc']:.3f}")
    auc = pd.DataFrame(rows)
    auc.round(4).to_csv(out / "cv_auc.csv", index=False)
    for task, title in TASKS.items():
        plots.auc_forest(auc[auc.task == task], out / f"auc_{task}.png", title)

    # --- final configuration per task: best model on "vessels + lesions" (chosen by AUC; only two candidates)
    ops, summary["final"] = [], {}
    for task, title in TASKS.items():
        sub = auc[(auc.task == task) & (auc.features == FINAL_SET)].sort_values("strat_auc", ascending=False).iloc[0]
        mname = sub["model"]; y = binary_target(df["grade"], task); X = df[FEATURE_SETS[FINAL_SET]].values.astype(float)
        proba, pred = cross_validate(X, y, MODELS[mname](), groups, 5, a.repeats, seed=0, target_sens=TARGET_SENS, strat=y + 2 * size_code)
        op = {"task": task, "model": mname, "features": FINAL_SET, "target_sensitivity": TARGET_SENS, **operating_point(y, pred),
              "positives": int(y.sum()), "negatives": int((1 - y).sum())}
        ops.append(op)
        mm = mixed_strata_mask(y, size.values)
        ops.append({**op, "features": FINAL_SET + " (only image sizes with both healthy and DR)", **operating_point(y[mm], pred[:, mm]),
                    "positives": int(y[mm].sum()), "negatives": int((1 - y[mm]).sum())})
        q = auc[(auc.task == task) & (auc.features == "image quality only")].sort_values("auc", ascending=False).iloc[0]
        qs = auc[(auc.task == task) & (auc.features == "image quality only")]["strat_auc"].max()
        plots.roc_curves({"Image quality only": (y, store[(task, "image quality only", q["model"])][1], q["auc"]),
                          "Vessels + lesions": (y, store[(task, FINAL_SET, mname)][1], sub["auc"])},
                         out / f"roc_{task}.png", f"{title}\n(cross-validated, {len(y)} images)")
        summary["final"][task] = {"model": mname, "auc": float(sub["auc"]), "auc_ci95": [float(sub["auc_ci95_low"]), float(sub["auc_ci95_high"])],
                                  "quality_only_auc": float(q["auc"]),
                                  "within_size_auc": float(sub["strat_auc"]), "within_size_auc_ci95": [float(sub["strat_auc_ci95_low"]), float(sub["strat_auc_ci95_high"])],
                                  "within_size_images_used": int(sub["strat_images_used"]), "quality_only_within_size_auc": float(qs)}
        if task == "any_dr" and a.perms:
            null = permutation_auc(X, y, MODELS[mname](), groups, a.perms)
            plots.permutation_hist(null, sub["auc"], out / "permutation_null.png")
            null_w = permutation_auc(X, y, MODELS[mname](), groups, a.perms, strata=size.values)
            summary["permutation_null"] = {"n": a.perms, "null_mean_auc": float(null.mean()), "null_max_auc": float(null.max()),
                                           "p_value": float((1 + (null >= sub["auc"]).sum()) / (1 + a.perms)),
                                           "within_size_null_mean_auc": float(null_w.mean()), "within_size_null_max_auc": float(null_w.max()),
                                           "within_size_p_value": float((1 + (null_w >= sub["strat_auc"]).sum()) / (1 + a.perms))}
    pd.DataFrame(ops).round(4).to_csv(out / "operating_point.csv", index=False)

    # --- one feature at a time: does it separate DR from healthy among images of the same size?
    y_any = binary_target(df["grade"], "any_dr")
    uni = pd.DataFrame([{"feature": c, "pooled_auc": roc_auc_score(y_any, df[c]),
                         "within_size_auc": stratified_auc(y_any, df[c].values, size.values)[0]}
                        for c in FEATURE_SETS["vessels + lesions + quality"]]).round(3)
    uni.sort_values("within_size_auc", key=lambda s: (s - 0.5).abs(), ascending=False).to_csv(out / "single_feature_auc.csv", index=False)

    # --- five grades
    five, cm = five_class(df, FEATURE_SETS[FINAL_SET], a.repeats, 0, mixed_strata_mask((df['grade'] > 0).astype(int).values, size.values))
    five.round(4).to_csv(out / "five_class.csv", index=False)
    plots.confusion(cm, [GRADES[i] for i in range(5)], out / "confusion_five_class.png", "5-grade XGBoost (cross-validated, one repeat)")

    # --- SHAP of the final any-DR model fitted on all images (interpretation only, not an accuracy estimate)
    if not a.no_shap:
        try:
            import shap
            task = "any_dr"; mname = summary["final"][task]["model"]
            cols = FEATURE_SETS[FINAL_SET]; X = df[cols].values.astype(float); y = binary_target(df["grade"], task)
            if "XGBoost" in mname:
                model = MODELS[mname]().set_params(scale_pos_weight=(y == 0).sum() / (y == 1).sum()).fit(X, y)
                sv = shap.TreeExplainer(model).shap_values(X)
            else:
                model = MODELS[mname]().fit(X, y)
                sv = shap.LinearExplainer(model[-1], model[:-1].transform(X)).shap_values(model[:-1].transform(X))
            plots.shap_summary(sv, X, cols, out / "shap_any_dr.png", f"What drives the 'any DR' prediction ({mname})")
        except ImportError:
            print("shap not installed; skipping the explanation figure")
    json.dump(summary, open(out / "metrics.json", "w"), indent=2)
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
