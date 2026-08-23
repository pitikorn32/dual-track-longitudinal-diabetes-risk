"""Cross-model feature-importance analysis for the dual-track paper.

Motivation
----------
The interpretability paragraph in the paper reports (a) model-agnostic univariate
effect sizes and (b) the GEE five-year winner's coefficients. A reviewer may ask
which features matter *for the other model families*, and whether the story is
consistent. This script answers that from the already-trained deployment
artifacts (no retraining), so the importances come from the exact models served
alongside the leaderboard.

For each saved model it produces a comparable global importance:
  * CatBoost  -> get_feature_importance() (PredictionValuesChange)
  * XGBoost   -> booster gain importance
  * LightGBM  -> booster gain importance
  * EBM       -> term_importances() (single-feature terms)
  * Logistic  -> |standardized coefficient|  (the stored coeffs are on the
                 standardized scale, so magnitudes are directly comparable)
One-hot columns are mapped back to their base feature and summed, then each
model's importances are normalized to shares (sum = 1) so families are compared
by *rank*, not by incompatible raw scales.

Outputs (written next to this script):
  feature_importance_by_model.csv   long table: one row per (model, base_feature)
  feature_importance_consensus.csv  base features ranked by cross-family agreement

Run from the submodule root:
    python feature_importance_analysis.py
"""
from __future__ import annotations

import re
from pathlib import Path

import numpy as np
import pandas as pd
import joblib

ROOT = Path(__file__).resolve().parent
MODELS_DIR = ROOT / "deployment" / "models"
PHASE1_DIR = ROOT / "digihealth_risk" / "phase_1" / "outputs"
OUT_BY_MODEL = ROOT / "feature_importance_by_model.csv"
OUT_CONSENSUS = ROOT / "feature_importance_consensus.csv"
TOPK = 10  # "top features" cutoff used for the agreement count

# The pure-prediction leaderboard winner (family) at each horizon, M=5 (Table I).
# The N=5 winner is GEE; its importance is read from the saved standardized (_z) GEE
# coefficients (phase_1 outputs), the other families from the trained model artifacts.
PURE_WINNER = {1: "catboost", 2: "logistic", 3: "xgboost", 4: "logistic", 5: "gee"}


def base_feature(name: str, categoricals: list[str]) -> str:
    """Map a transformed (possibly one-hot) column back to its base feature."""
    for c in sorted(categoricals, key=len, reverse=True):
        if name == c or name.startswith(c + "_"):
            return c
    return name


def raw_importance(art: dict) -> np.ndarray | None:
    """Return a per-transformed-feature importance vector for one artifact."""
    fam = art.get("model_family", "").lower()
    names = art["transformed_feature_names"]
    n = len(names)
    model = art.get("model")
    try:
        if fam == "catboost" and model is not None:
            imp = np.asarray(model.get_feature_importance(), dtype=float)
        elif fam == "xgboost" and model is not None:
            booster = model.get_booster()
            score = booster.get_score(importance_type="gain")  # {f0: gain}
            imp = np.array([score.get(f"f{i}", 0.0) for i in range(n)], dtype=float)
        elif fam == "lightgbm" and model is not None:
            imp = np.asarray(model.booster_.feature_importance(importance_type="gain"), dtype=float)
        elif fam == "ebm" and model is not None:
            ti = np.asarray(model.term_importances(), dtype=float)
            names_ebm = list(getattr(model, "term_names_", []))
            positional = names_ebm[:n] == [f"feature_{i:04d}" for i in range(n)]
            if positional and len(ti) >= n:
                imp = ti[:n]              # EBM single-feature terms in column order
            else:
                vec = np.zeros(n)
                idx = {t: i for i, t in enumerate(names)}
                for imp_v, tname in zip(ti, names_ebm):
                    if tname in idx:
                        vec[idx[tname]] = imp_v
                imp = vec
        elif "coefficients" in art:       # logistic: [intercept, *standardized betas]
            coefs = np.asarray(art["coefficients"], dtype=float)
            imp = np.abs(coefs[1:]) if len(coefs) == n + 1 else np.abs(coefs[:n])
        else:
            return None
    except Exception as e:  # noqa: BLE001
        print(f"  ! importance failed for {art.get('model_key')}: {e!r}")
        return None
    if imp.shape[0] != n:
        print(f"  ! length mismatch for {art.get('model_key')}: {imp.shape[0]} vs {n}")
        return None
    return imp


def per_base_shares(art: dict, imp: np.ndarray) -> pd.Series:
    names = art["transformed_feature_names"]
    cats = art.get("categorical_features", [])
    bases = [base_feature(x, cats) for x in names]
    s = pd.Series(imp, index=bases).groupby(level=0).sum()
    total = s.sum()
    return (s / total) if total > 0 else s


def stat_base(name: str) -> str:
    """Base feature for a statistical coefficient name (strip _z, unwrap C(var)[T.k])."""
    m = re.match(r"C\((\w+)\)\[", name)
    if m:
        return m.group(1)
    return re.sub(r"_z$", "", name)


def gee_shares(n: int, m: int = 5) -> pd.Series | None:
    """Standardized-|coef| importance shares for the GEE model at (n, m)."""
    f = PHASE1_DIR / f"phase_1_v2_gee_horizon_{n}_history_{m}_coefficients.csv"
    if not f.exists():
        return None
    d = pd.read_csv(f)
    d = d[~d["feature"].str.lower().eq("intercept")].copy()
    d["base"] = d["feature"].map(stat_base)
    s = d.assign(imp=d["coefficient"].abs()).groupby("base")["imp"].sum()
    total = s.sum()
    return (s / total) if total > 0 else s


def main() -> None:
    paths = sorted(MODELS_DIR.glob("*.joblib"))
    if not paths:
        raise SystemExit(f"No model artifacts in {MODELS_DIR}")
    rows = []
    for p in paths:
        art = joblib.load(p)
        imp = raw_importance(art)
        if imp is None:
            continue
        shares = per_base_shares(art, imp).sort_values(ascending=False)
        ranks = shares.rank(ascending=False, method="min")
        for feat, share in shares.items():
            rows.append(dict(
                model_key=art.get("model_key"), track=art.get("track"),
                family=art.get("model_family"), horizon=art.get("horizon_years"),
                history=art.get("history_years"), base_feature=feat,
                importance_share=round(float(share), 5), rank=int(ranks[feat]),
            ))
    # GEE is the N=5 pure-prediction winner; add it (and the other horizons, as a
    # statistical comparator) from its saved standardized coefficients.
    for n in [1, 2, 3, 4, 5]:
        sh = gee_shares(n, 5)
        if sh is None:
            continue
        sh = sh.sort_values(ascending=False)
        rk = sh.rank(ascending=False, method="min")
        for feat, share in sh.items():
            rows.append(dict(
                model_key=f"gee_n{n}_m5", track="leaderboard", family="gee",
                horizon=n, history=5, base_feature=feat,
                importance_share=round(float(share), 5), rank=int(rk[feat]),
            ))

    long = pd.DataFrame(rows)
    long.to_csv(OUT_BY_MODEL, index=False)
    print(f"Wrote {OUT_BY_MODEL.name}: {len(long)} rows, "
          f"{long['model_key'].nunique()} models, families={sorted(long['family'].unique())}")

    def show(df, title, k=12):
        print(f"\n=== {title} ===")
        with pd.option_context("display.max_rows", None, "display.width", 130):
            print(df.head(k).to_string())

    # Restrict to the M=5 leaderboard finalists (Table I is all M=5).
    fin = long[long.history == 5].copy()

    # ---- (B0) Headline: consensus over the TRUE pure-prediction winners (Table I) ----
    win_keys = []
    for N, fam in PURE_WINNER.items():
        cand = fin[(fin.horizon == N) & (fin.family == fam)
                   & (fin.track.isin(["screening", "leaderboard"]))]
        if not cand.empty:
            win_keys.append(cand.model_key.iloc[0])
    win = fin[fin.model_key.isin(win_keys)].copy()
    win["in_topk"] = win["rank"] <= TOPK
    winc = (win.groupby("base_feature")
            .agg(mean_share=("importance_share", "mean"), mean_rank=("rank", "mean"),
                 topk_hits=("in_topk", "sum"), n_families=("family", "nunique"))
            .sort_values(["topk_hits", "mean_share"], ascending=[False, False]))
    print("\nPure-prediction leaderboard winners (M=5):", win_keys)
    for mk in win_keys:
        d = win[win.model_key == mk].nlargest(6, "importance_share")
        feats = ", ".join(f"{r.base_feature} ({r.importance_share:.2f})" for r in d.itertuples())
        print(f"  {d.family.iloc[0]:9s} N={d.horizon.iloc[0]}: {feats}")
    show(winc, f"Consensus over pure-prediction winners (families={sorted(win.family.unique())})", k=12)

    # ---- (A) Per-horizon winner top features, both tracks ----
    for track in ["screening", "intervention"]:
        print(f"\n----- {track} track, M=5 winners -----")
        d0 = fin[fin.track == track]
        for N in sorted(d0.horizon.unique()):
            d = d0[d0.horizon == N].nlargest(8, "importance_share")
            fam = d.family.iloc[0]
            feats = ", ".join(f"{r.base_feature} ({r.importance_share:.2f})" for r in d.itertuples())
            print(f"N={N} [{fam:8s}]: {feats}")

    # ---- (B) Cross-family consensus over all M=5 winners (both tracks) ----
    fin["in_topk"] = fin["rank"] <= TOPK
    consensus = (fin.groupby("base_feature")
                 .agg(mean_share=("importance_share", "mean"),
                      mean_rank=("rank", "mean"),
                      topk_hits=("in_topk", "sum"),
                      n_models=("model_key", "nunique"),
                      n_families=("family", "nunique"))
                 .sort_values(["topk_hits", "mean_share"], ascending=[False, False]))
    consensus.to_csv(OUT_CONSENSUS)
    show(consensus, f"Cross-family consensus, M=5 winners "
                    f"(families={sorted(fin.family.unique())}, {fin.model_key.nunique()} models)", k=15)

    # ---- (C) Same-setting head-to-head: N=5, M=5, statistical vs tree ----
    pair = fin[(fin.horizon == 5)]
    piv = (pair.pivot_table(index="base_feature", columns="family",
                            values="importance_share", aggfunc="mean").fillna(0.0))
    piv["max"] = piv.max(axis=1)
    show(piv.sort_values("max", ascending=False).drop(columns="max"),
         "N=5, M=5 head-to-head: screening logistic vs intervention catboost (importance share)")


if __name__ == "__main__":
    main()
