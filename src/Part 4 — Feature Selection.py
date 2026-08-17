import os
import json
import numpy as np
import pandas as pd
from scipy import stats
from scipy.stats import pearsonr, spearmanr
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import ElasticNet, Lasso, LinearRegression
from sklearn.feature_selection import mutual_info_regression, f_regression, RFE, SequentialFeatureSelector
from sklearn.inspection import permutation_importance
from setup_imports import DATA_DIR

def compute_vif(X, j):
    y_j = X[:, j]
    X_others = np.column_stack([np.ones(len(y_j)), np.delete(X, j, axis=1)])
    beta = np.linalg.lstsq(X_others, y_j, rcond=None)[0]
    r2 = 1 - np.sum((y_j - X_others @ beta) ** 2) / (
        np.sum((y_j - y_j.mean()) ** 2) + 1e-10
    )
    return 1.0 / (1.0 - min(max(r2, 0), 0.9999))


def granger_pvalue(y, x, maxlag=8):
    n = len(y)
    min_p = 1.0
    for lag in range(1, maxlag + 1):
        if n <= 2 * lag + 5:
            continue
        ne = n - lag
        Y = y[lag:]
        Yr = np.column_stack(
            [np.ones(ne)] + [y[lag - k - 1 : n - k - 1] for k in range(lag)]
        )
        Yu = np.column_stack([Yr] + [x[lag - k - 1 : n - k - 1] for k in range(lag)])
        try:
            br = np.linalg.lstsq(Yr, Y, rcond=None)[0]
            bu = np.linalg.lstsq(Yu, Y, rcond=None)[0]
            rr = np.sum((Y - Yr @ br) ** 2)
            ru = np.sum((Y - Yu @ bu) ** 2)
            denom = ne - 2 * lag - 1
            if denom <= 0 or ru <= 0:
                continue
            F = ((rr - ru) / lag) / (ru / denom)
            min_p = min(min_p, float(1 - stats.f.cdf(F, lag, denom)))
        except Exception:
            pass
    return min_p


def run_feature_selection(X, y, X_test, y_test, df_train, features):
    print("Running 13 feature-selection methods...\n")
    votes = {f: 0 for f in features}
    scores = {}

    print("  [VIF] removing multicollinear features...")
    remaining = list(features)
    removed = []
    while True:
        Xs = np.column_stack([X[:, features.index(f)] for f in remaining])
        vifs = {f: compute_vif(Xs, i) for i, f in enumerate(remaining)}
        worst = max(vifs, key=vifs.get)
        if vifs[worst] > 10 and len(remaining) > 5:
            remaining.remove(worst)
            removed.append(worst)
        else:
            scores["VIF"] = vifs
            break
    print(f"        removed {len(removed)}: {removed}")

    keep_mask = np.array([f in remaining for f in features])
    Xv = X[:, keep_mask]
    Xtv = X_test[:, keep_mask]
    fv = [f for f in features if f in remaining]

    sc = {f: abs(pearsonr(Xv[:, i], y)[0]) for i, f in enumerate(fv)}
    scores["Pearson"] = sc
    for f in [f for f, v in sc.items() if v > 0.15]:
        votes[f] += 1
    print("  [1] Pearson done")

    sc = {f: abs(spearmanr(Xv[:, i], y)[0]) for i, f in enumerate(fv)}
    scores["Spearman"] = sc
    for f in [f for f, v in sc.items() if v > 0.15]:
        votes[f] += 1
    print("  [2] Spearman done")

    mi = mutual_info_regression(Xv, y, random_state=42)
    sc = dict(zip(fv, mi))
    scores["MutualInfo"] = sc
    for f in [f for f, v in sc.items() if v > 0.05]:
        votes[f] += 1
    print("  [3] Mutual Information done")

    fvals, pvals = f_regression(Xv, y)
    scores["F-stat"] = dict(zip(fv, fvals))
    for f in [f for f, p in zip(fv, pvals) if p < 0.05]:
        votes[f] += 1
    print("  [4] F-statistic done")

    rfe = RFE(LinearRegression(), n_features_to_select=min(15, len(fv)))
    rfe.fit(Xv, y)
    scores["RFE"] = {f: 1 / r for f, r in zip(fv, rfe.ranking_)}
    for f in [f for f, s in zip(fv, rfe.support_) if s]:
        votes[f] += 1
    print("  [5] RFE done")

    n_sub = min(4000, len(y))
    idx = np.random.choice(len(y), n_sub, replace=False)
    sfs = SequentialFeatureSelector(
        LinearRegression(),
        n_features_to_select=min(15, len(fv)),
        direction="forward",
        cv=3,
        n_jobs=-1,
    )
    sfs.fit(Xv[idx], y[idx])
    sel_sfs = set(np.array(fv)[sfs.get_support()])
    scores["SFS"] = {f: (1 if f in sel_sfs else 0) for f in fv}
    for f in sel_sfs:
        votes[f] += 1
    print("  [6] Sequential Selection done")

    best_mae, best_coef = np.inf, None
    for a in [0.001, 0.01, 0.05, 0.1, 0.5, 1.0]:
        m = Lasso(alpha=a, max_iter=5000).fit(Xv, y)
        mae = np.mean(np.abs(y - m.predict(Xv)))
        if mae < best_mae and np.sum(m.coef_ != 0) >= 5:
            best_mae, best_coef = mae, m.coef_.copy()
    sc = {f: abs(c) for f, c in zip(fv, best_coef)}
    scores["LASSO"] = sc
    for f in [f for f, v in sc.items() if v > 0]:
        votes[f] += 1
    print("  [7] LASSO done")

    best_mae, best_coef = np.inf, None
    for a in [0.01, 0.1, 0.5]:
        for l1 in [0.3, 0.5, 0.7]:
            m = ElasticNet(alpha=a, l1_ratio=l1, max_iter=5000).fit(Xv, y)
            mae = np.mean(np.abs(y - m.predict(Xv)))
            if mae < best_mae and np.sum(m.coef_ != 0) >= 5:
                best_mae, best_coef = mae, m.coef_.copy()
    sc = {f: abs(c) for f, c in zip(fv, best_coef)}
    scores["ElasticNet"] = sc
    for f in [f for f, v in sc.items() if v > 0]:
        votes[f] += 1
    print("  [8] Elastic Net done")

    rf = RandomForestRegressor(
        n_estimators=100,
        max_depth=8,
        min_samples_leaf=20,
        random_state=42,
        n_jobs=-1,
    )
    rf.fit(Xv, y)
    sc = dict(zip(fv, rf.feature_importances_))
    scores["RF"] = sc
    for f in [f for f, v in sc.items() if v > 0.01]:
        votes[f] += 1
    print("  [9] Random Forest done")

    n_p = min(2000, len(X_test))
    idx_p = np.random.choice(len(X_test), n_p, replace=False)
    perm = permutation_importance(
        rf, Xtv[idx_p], y_test[idx_p], n_repeats=5, random_state=42
    )
    sc = dict(zip(fv, perm.importances_mean))
    scores["Permutation"] = sc
    for f in [f for f, v in sc.items() if v > 0.001]:
        votes[f] += 1
    print("  [10] Permutation done")

    sc = {}
    for f in fv:
        if f in [
            "hour_sin", "hour_cos", "dow_sin", "dow_cos", "month_sin", "month_cos",
        ]:
            sc[f] = 0.5
        else:
            x_arr = (
                df_train[f].values
                if f in df_train.columns
                else Xv[:, fv.index(f)]
            )
            sc[f] = 1.0 - granger_pvalue(df_train["price"].values, x_arr)
    scores["Granger"] = sc
    for f in [f for f, v in sc.items() if v > 0.95]:
        votes[f] += 1
    print("  [11] Granger causality done")

    base = rf.predict(Xtv[:500])
    shap = np.zeros(len(fv))
    for j in range(len(fv)):
        Xp = Xtv[:500].copy()
        Xp[:, j] = np.random.permutation(Xp[:, j])
        shap[j] = np.mean(np.abs(base - rf.predict(Xp)))
    sc = dict(zip(fv, shap))
    scores["SHAP"] = sc
    mx = max(sc.values()) or 1
    for f in [f for f, v in sc.items() if v > 0.1 * mx]:
        votes[f] += 1
    print("  [12] SHAP done")

    for f in removed:
        votes[f] = 0

    selected = [f for f, v in votes.items() if v >= 5]
    print(f"\n  RESULT: {len(selected)} features selected (>= 5 votes)")
    print(f"  {'Feature':<22}{'Votes':>6}  Decision")
    print("  " + "-" * 42)
    for f, v in sorted(votes.items(), key=lambda x: -x[1]):
        print(f"  {f:<22}{v:>6}  {'SELECT' if v >= 5 else 'drop'}")

    return selected, scores, votes, removed


if __name__ == "__main__":
    df = pd.read_pickle(os.path.join(DATA_DIR, "data_part2.pkl"))
    with open(os.path.join(DATA_DIR, "features_part2.json")) as f:
        features = json.load(f)

    train = df[df["datetime"] <= "2024-04-30 23:00"].copy()
    val = df[
        (df["datetime"] >= "2024-05-01") & (df["datetime"] <= "2025-04-30 23:00")
    ].copy()
    test = df[df["datetime"] >= "2025-05-01"].copy()

    scaler_x = StandardScaler()
    scaler_y = StandardScaler()

    X_tr = scaler_x.fit_transform(train[features].values)
    y_tr = scaler_y.fit_transform(train["price"].values.reshape(-1, 1)).flatten()

    X_val = scaler_x.transform(val[features].values)
    y_val = val["price"].values

    X_te = scaler_x.transform(test[features].values)
    y_te = test["price"].values

    selected, scores, votes, removed = run_feature_selection(
        X_tr, y_tr, X_te, y_te, train, features
    )

    np.save(
        os.path.join(DATA_DIR, "fs_part4.npy"),
        {
            "selected": selected,
            "scores": scores,
            "votes": votes,
            "removed": removed,
            "features": features,
        },
        allow_pickle=True,
    )
    with open(os.path.join(DATA_DIR, "selected_part4.json"), "w") as f:
        json.dump(selected, f)

    print("\n✓ Saved feature selection — ready for Part 5")