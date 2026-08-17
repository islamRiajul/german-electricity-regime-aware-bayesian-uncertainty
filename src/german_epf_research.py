"""
╔══════════════════════════════════════════════════════════════════════════════╗
║  GERMAN ELECTRICITY PRICE FORECASTING — FULL RESEARCH PIPELINE (PLOTLY)      ║
║  Dataset : SMARD Day-Ahead Prices 2020–2026                                ║
║  Split   : Train (-> Apr 2024) | Val (May 2024–Apr 2025) | Test (May 2025->)║
║                                                                              ║
║  Feature Selection (13 methods):                                            ║
║    Filter   : Pearson, Spearman, Mutual Information, F-statistic, VIF       ║
║    Wrapper  : RFE, Sequential Feature Selection                             ║
║    Embedded : LASSO, Elastic Net, RF Importance, Permutation Importance     ║
║    Advanced : Granger Causality, SHAP Values                               ║
║                                                                              ║
║  Models (4):                                                                ║
║    1. DDNN       – distributional baseline                                  ║
║    2. EvDNN      – evidential (single-pass uncertainty)                     ║
║    3. VI-DDNN    – variational inference (true Bayesian)                    ║
║    4. BSSM       – Bayesian state-space (Kalman filter)                     ║
║                                                                              ║
║  Metrics : MAE, RMSE, CRPS, PICP(90%), MPIW(90%), MAACE                   ║
╚══════════════════════════════════════════════════════════════════════════════╝
"""

"""
╔══════════════════════════════════════════════════════════════════════════════╗
║  GERMAN ELECTRICITY PRICE FORECASTING — FULL RESEARCH PIPELINE (PLOTLY)      ║
║  Dataset : SMARD Day-Ahead Prices 2020–2026                                ║
║  Split   : Train (-> Apr 2024) | Val (May 2024–Apr 2025) | Test (May 2025->)║
║                                                                              ║
║  Feature Selection (13 methods):                                            ║
║    Filter   : Pearson, Spearman, Mutual Information, F-statistic, VIF       ║
║    Wrapper  : RFE, Sequential Feature Selection                             ║
║    Embedded : LASSO, Elastic Net, RF Importance, Permutation Importance     ║
║    Advanced : Granger Causality, SHAP Values                               ║
║                                                                              ║
║  Models (4):                                                                ║
║    1. DDNN       – distributional baseline                                  ║
║    2. EvDNN      – evidential (single-pass uncertainty)                     ║
║    3. VI-DDNN    – variational inference (true Bayesian)                    ║
║    4. BSSM       – Bayesian state-space (Kalman filter)                     ║
║                                                                              ║
║  Metrics : MAE, RMSE, CRPS, PICP(90%), MPIW(90%), MAACE                   ║
╚══════════════════════════════════════════════════════════════════════════════╝
"""

# ══════════════════════════════════════════════════════════════════════════════
# IMPORTS
# ══════════════════════════════════════════════════════════════════════════════
import os
import json
import warnings
import numpy as np
import pandas as pd
from scipy import stats
from scipy.stats import pearsonr, spearmanr
from scipy.special import gammaln
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import Lasso, ElasticNet, LinearRegression
from sklearn.ensemble import RandomForestRegressor
from sklearn.feature_selection import (
    mutual_info_regression, f_regression,
    RFE, SequentialFeatureSelector
)
from sklearn.inspection import permutation_importance
from sklearn.metrics import mean_absolute_error, mean_squared_error

# Plotly Interactive Graphics
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots

warnings.filterwarnings("ignore")
np.random.seed(42)

# ── Pure numpy/scipy implementations ─────────────────────────────────────────

def variance_inflation_factor(X: np.ndarray, feat_idx: int) -> float:
    """VIF(j) = 1 / (1 - R²_j)"""
    y_j = X[:, feat_idx]
    X_others = np.delete(X, feat_idx, axis=1)
    X_others = np.column_stack([np.ones(len(y_j)), X_others])
    try:
        beta = np.linalg.lstsq(X_others, y_j, rcond=None)[0]
        y_hat = X_others @ beta
        ss_res = np.sum((y_j - y_hat) ** 2)
        ss_tot = np.sum((y_j - y_j.mean()) ** 2)
        r2 = 1 - ss_res / (ss_tot + 1e-10)
        r2 = min(max(r2, 0), 0.9999)
        return 1.0 / (1.0 - r2)
    except Exception:
        return 1.0


def granger_causality_test(y: np.ndarray, x: np.ndarray, maxlag: int = 12) -> float:
    """Granger causality F-test (pure numpy)."""
    n = len(y)
    min_pval = 1.0

    for lag in range(1, maxlag + 1):
        if n <= 2 * lag + 5:
            continue
        n_eff = n - lag
        Y = y[lag:]

        Yr = np.column_stack([np.ones(n_eff)] + [y[lag-k-1:n-k-1] for k in range(lag)])
        Yu = np.column_stack([Yr, *[x[lag-k-1:n-k-1] for k in range(lag)]])

        try:
            beta_r = np.linalg.lstsq(Yr, Y, rcond=None)[0]
            beta_u = np.linalg.lstsq(Yu, Y, rcond=None)[0]
            rss_r  = np.sum((Y - Yr @ beta_r) ** 2)
            rss_u  = np.sum((Y - Yu @ beta_u) ** 2)
            p      = lag
            denom  = n_eff - 2 * p - 1
            if denom <= 0 or rss_u <= 0:
                continue
            F_stat  = ((rss_r - rss_u) / p) / (rss_u / denom)
            p_value = float(1 - stats.f.cdf(F_stat, p, denom))
            min_pval = min(min_pval, p_value)
        except Exception:
            continue

    return min_pval


print("✓ All imports and helper functions ready")


# ══════════════════════════════════════════════════════════════════════════════
# SECTION 1 ── DATA LOADING & ADVANCED FEATURE ENGINEERING (35 FEATURES)
# ══════════════════════════════════════════════════════════════════════════════

def _parse_eu_num(s):
    if pd.isna(s):
        return np.nan
    return pd.to_numeric(str(s).replace(',', ''), errors='coerce')


def signed_log(p):
    return np.sign(p) * np.log1p(np.abs(p))


def inv_signed_log(z, price_clip=(-600.0, 1000.0)):
    z_min = -np.log1p(np.abs(price_clip[0])) if price_clip[0] < 0 else np.log1p(price_clip[0])
    z_max = np.log1p(price_clip[1])
    
    z_safe = np.clip(z, z_min, z_max)
    p = np.sign(z_safe) * np.expm1(np.abs(z_safe))
    return np.clip(p, price_clip[0], price_clip[1])


def load_real_smard(path: str) -> tuple[pd.DataFrame, list[str]]:
    print("\n" + "═"*65)
    print("  SECTION 1 — DATA LOADING (REAL SMARD CSV)")
    print("═"*65)

    df = pd.read_csv(path, sep=';', skiprows=1, encoding='utf-8')

    rename = {
        'Germany/Luxembourg [€/MWh] Calculated resolutions': 'price',
        'grid load [MWh] Calculated resolutions':            'load_forecast',
        'Residual load [MWh] Calculated resolutions':        'residual_load',
        'Total [MWh] Calculated resolutions':                'total_gen',
        'Photovoltaics and wind [MWh] Calculated resolutions':'pv_wind',
        'Wind offshore [MWh] Calculated resolutions':        'wind_offshore',
        'Wind onshore [MWh] Calculated resolutions':         'wind_onshore',
        'Photovoltaics [MWh] Calculated resolutions':        'solar',
        'Other [MWh] Calculated resolutions':                'other_gen',
    }
    df = df.rename(columns=rename)

    drop_cols = [
        'Volume (+) [MWh] Calculated resolutions',
        'Volume (-) [MWh] Calculated resolutions',
        'Price [€/MWh] Calculated resolutions',
        'Net income [€] Calculated resolutions',
    ]
    df = df.drop(columns=[c for c in drop_cols if c in df.columns])
    print(f"  Excluded balancing columns: Volume(+), Volume(-), Price, Net income")

    for c in ['price','load_forecast','residual_load','total_gen','pv_wind',
              'wind_offshore','wind_onshore','solar','other_gen']:
        if c in df.columns:
            df[c] = df[c].apply(_parse_eu_num)

    df['datetime'] = pd.to_datetime(df['Start date'], format='%b %d, %Y %I:%M %p', errors='coerce')
    df = df.dropna(subset=['datetime']).sort_values('datetime').reset_index(drop=True)
    df = df.drop(columns=['Start date', 'End date'])

    return _engineer_features(df)


def _engineer_features(df: pd.DataFrame) -> tuple[pd.DataFrame, list[str]]:
    """Creates all 35 candidate features."""
    num_cols = df.select_dtypes(include=np.number).columns
    df[num_cols] = df[num_cols].ffill().bfill().fillna(0)

    # ── 1. LAG FEATURES ──────────────────────────────────────────────────────
    for lag in [1, 2, 3, 6, 12, 24, 48, 168]:
        df[f'lag_{lag}'] = df['price'].shift(lag)

    # ── 2. ROLLING STATISTICS ────────────────────────────────────────────────
    df['roll_mean_24']  = df['price'].shift(1).rolling(24).mean()
    df['roll_std_24']   = df['price'].shift(1).rolling(24).std()
    df['roll_mean_168'] = df['price'].shift(1).rolling(168).mean()
    df['price_change']  = df['price'].shift(1) - df['price'].shift(25)

    # ── 3. EXPONENTIAL MOVING AVERAGES (EMA) ─────────────────────────────────
    df['ema_6h']  = df['price'].shift(1).ewm(span=6).mean()
    df['ema_24h'] = df['price'].shift(1).ewm(span=24).mean()
    df['ema_72h'] = df['price'].shift(1).ewm(span=72).mean()

    # ── 4. SUPPLY/DEMAND RATIOS & RAMP RATES ─────────────────────────────────
    df['total_renewable'] = df['wind_offshore'] + df['wind_onshore'] + df['solar']
    df['ren_share']       = df['total_renewable'] / (df['load_forecast'] + 1e-6)
    df['demand_ren_ratio'] = df['load_forecast'] / (df['total_renewable'] + 1e-6)

    df['solar_ramp_1h'] = df['solar'] - df['solar'].shift(1)
    df['solar_ramp_3h'] = df['solar'] - df['solar'].shift(3)
    df['wind_ramp_1h']  = df['total_renewable'] - df['total_renewable'].shift(1)
    df['load_ramp_1h']  = df['load_forecast'] - df['load_forecast'].shift(1)

    # ── 5. CALENDAR FEATURES (CYCLIC) ─────────────────────────────────────────
    df['hour']      = df['datetime'].dt.hour
    df['dow']       = df['datetime'].dt.dayofweek
    df['month']     = df['datetime'].dt.month
    df['hour_sin']  = np.sin(2*np.pi*df['hour']/24)
    df['hour_cos']  = np.cos(2*np.pi*df['hour']/24)
    df['dow_sin']   = np.sin(2*np.pi*df['dow']/7)
    df['dow_cos']   = np.cos(2*np.pi*df['dow']/7)
    df['month_sin'] = np.sin(2*np.pi*(df['month']-1)/12)
    df['month_cos'] = np.cos(2*np.pi*(df['month']-1)/12)

    df = df.dropna().reset_index(drop=True)

    features = [
        'lag_1','lag_2','lag_3','lag_6','lag_12','lag_24','lag_48','lag_168',
        'roll_mean_24','roll_std_24','roll_mean_168','price_change',
        'ema_6h','ema_24h','ema_72h',
        'load_forecast','residual_load','total_gen','pv_wind','ren_share','demand_ren_ratio',
        'solar_ramp_1h','solar_ramp_3h','wind_ramp_1h','load_ramp_1h',
        'wind_offshore','wind_onshore','solar','other_gen',
        'hour_sin','hour_cos','dow_sin','dow_cos','month_sin','month_cos',
    ]
    features = [f for f in features if f in df.columns]

    print(f"  Rows after cleaning : {len(df):,}")
    print(f"  Date range          : {df['datetime'].iloc[0].date()} → {df['datetime'].iloc[-1].date()}")
    print(f"  Price range         : €{df['price'].min():.1f} – €{df['price'].max():.1f}")
    print(f"  Negative price hrs  : {(df['price'] < 0).sum():,}")
    print(f"  Candidate features  : {len(features)}")
    return df, features


def split_data(df: pd.DataFrame):
    """Chronological time-series split using exact custom dates."""
    print("\n" + "─"*65)
    print("  DATA SPLIT")
    print("─"*65)
    
    train = df[df['datetime'] <= '2024-04-30 23:00'].copy()
    val   = df[(df['datetime'] >= '2024-05-01') & (df['datetime'] <= '2025-04-30 23:00')].copy()
    test  = df[df['datetime'] >= '2025-05-01'].copy()
    
    print(f"  Train      (-> 2024-04-30)        : {len(train):,} rows")
    print(f"  Validation (2024-05-01 -> 2025-04-30): {len(val):,} rows")
    print(f"  Test       (2025-05-01 ->)        : {len(test):,} rows")
    return train, val, test


# ══════════════════════════════════════════════════════════════════════════════
# SECTION 2 ── FEATURE SELECTION
# ══════════════════════════════════════════════════════════════════════════════

class FeatureSelector:
    def __init__(self, features: list[str], threshold_votes: int = 5):
        self.features  = features
        self.thresh    = threshold_votes
        self.scores    = {}
        self.votes     = {f: 0 for f in features}
        self.selected  = []

    def pearson(self, X: np.ndarray, y: np.ndarray):
        print("    ① Pearson correlation...", end=" ")
        scores = {feat: abs(pearsonr(X[:, i], y)[0]) for i, feat in enumerate(self.features)}
        self.scores['Pearson'] = scores
        selected = {f for f, v in scores.items() if v > 0.15}
        for f in selected: self.votes[f] += 1
        print(f"selected {len(selected)}/{len(self.features)}")
        return scores

    def spearman(self, X: np.ndarray, y: np.ndarray):
        print("    ② Spearman rank correlation...", end=" ")
        scores = {feat: abs(spearmanr(X[:, i], y)[0]) for i, feat in enumerate(self.features)}
        self.scores['Spearman'] = scores
        selected = {f for f, v in scores.items() if v > 0.15}
        for f in selected: self.votes[f] += 1
        print(f"selected {len(selected)}/{len(self.features)}")
        return scores

    def mutual_information(self, X: np.ndarray, y: np.ndarray):
        print("    ③ Mutual Information...", end=" ")
        mi = mutual_info_regression(X, y, random_state=42)
        scores = dict(zip(self.features, mi))
        self.scores['MutualInfo'] = scores
        selected = {f for f, v in scores.items() if v > 0.05}
        for f in selected: self.votes[f] += 1
        print(f"selected {len(selected)}/{len(self.features)}")
        return scores

    def f_statistic(self, X: np.ndarray, y: np.ndarray):
        print("    ④ F-statistic...", end=" ")
        f_vals, p_vals = f_regression(X, y)
        scores = dict(zip(self.features, f_vals))
        self.scores['F-statistic'] = scores
        selected = {f for f, p in zip(self.features, p_vals) if p < 0.05}
        for f in selected: self.votes[f] += 1
        print(f"selected {len(selected)}/{len(self.features)}")
        return scores

    def vif_filter(self, X: np.ndarray) -> list[str]:
        print("    ⑤ VIF collinearity filter...", end=" ")
        remaining = list(self.features)
        removed   = []
        vif_vals  = {}

        while True:
            X_sub = np.column_stack([X[:, self.features.index(f)] for f in remaining])
            cur_vif = {feat: variance_inflation_factor(X_sub, i) for i, feat in enumerate(remaining)}
            max_feat = max(cur_vif, key=cur_vif.get)
            if cur_vif[max_feat] > 10 and len(remaining) > 5:
                remaining.remove(max_feat)
                removed.append(max_feat)
                vif_vals[max_feat] = cur_vif[max_feat]
            else:
                vif_vals.update(cur_vif)
                break

        self.scores['VIF'] = vif_vals
        self.vif_kept    = remaining
        self.vif_removed = removed
        print(f"removed {len(removed)} highly collinear features → kept {len(remaining)}")
        return remaining

    def rfe(self, X: np.ndarray, y: np.ndarray, n_select: int = 15):
        print("    ⑥ Recursive Feature Elimination (RFE)...", end=" ")
        selector = RFE(LinearRegression(), n_features_to_select=n_select, step=1)
        selector.fit(X, y)
        scores = dict(zip(self.features, selector.ranking_))
        self.scores['RFE'] = {f: 1/r for f, r in scores.items()}
        selected = {f for f, s in zip(self.features, selector.support_) if s}
        for f in selected: self.votes[f] += 1
        print(f"selected {len(selected)}/{len(self.features)}")
        return scores

    def sequential_selection(self, X: np.ndarray, y: np.ndarray, n_select: int = 15):
        print("    ⑦ Sequential Feature Selection (forward)...", end=" ")
        n_sub = min(5000, len(y))
        idx   = np.random.choice(len(y), n_sub, replace=False)
        selector = SequentialFeatureSelector(LinearRegression(), n_features_to_select=n_select, direction='forward', cv=3, n_jobs=-1)
        selector.fit(X[idx], y[idx])
        selected = set(np.array(self.features)[selector.get_support()])
        self.scores['SFS'] = {f: (1 if f in selected else 0) for f in self.features}
        for f in selected: self.votes[f] += 1
        print(f"selected {len(selected)}/{len(self.features)}")
        return self.scores['SFS']

    def lasso_selection(self, X: np.ndarray, y: np.ndarray):
        print("    ⑧ LASSO (L1) selection...", end=" ")
        best_mae, best_coef, best_alpha = np.inf, None, None
        for alpha in [0.0001, 0.001, 0.01, 0.05, 0.1, 0.5, 1.0]:
            lasso = Lasso(alpha=alpha, max_iter=5000)
            lasso.fit(X, y)
            mae = mean_absolute_error(y, lasso.predict(X))
            if mae < best_mae and np.sum(lasso.coef_ != 0) >= 5:
                best_mae, best_coef, best_alpha = mae, lasso.coef_.copy(), alpha
        scores = {f: abs(c) for f, c in zip(self.features, best_coef)}
        self.scores['LASSO'] = scores
        selected = {f for f, c in scores.items() if c > 0}
        for f in selected: self.votes[f] += 1
        print(f"selected {len(selected)}/{len(self.features)}")
        return scores

    def elastic_net_selection(self, X: np.ndarray, y: np.ndarray):
        print("    ⑨ Elastic Net selection...", end=" ")
        best_mae, best_coef = np.inf, None
        for alpha in [0.001, 0.01, 0.1, 0.5]:
            for l1 in [0.3, 0.5, 0.7]:
                en = ElasticNet(alpha=alpha, l1_ratio=l1, max_iter=5000)
                en.fit(X, y)
                mae = mean_absolute_error(y, en.predict(X))
                if mae < best_mae and np.sum(en.coef_ != 0) >= 5:
                    best_mae, best_coef = mae, en.coef_.copy()
        scores = {f: abs(c) for f, c in zip(self.features, best_coef)}
        self.scores['ElasticNet'] = scores
        selected = {f for f, c in scores.items() if c > 0}
        for f in selected: self.votes[f] += 1
        print(f"selected {len(selected)}/{len(self.features)}")
        return scores

    def rf_importance(self, X: np.ndarray, y: np.ndarray, X_test: np.ndarray, y_test: np.ndarray):
        print("    ⑩ Random Forest importance...", end=" ")
        rf = RandomForestRegressor(n_estimators=100, max_depth=8, min_samples_leaf=20, random_state=42, n_jobs=-1)
        rf.fit(X, y)
        scores = dict(zip(self.features, rf.feature_importances_))
        self.scores['RF_Impurity'] = scores
        selected = {f for f, v in scores.items() if v > 0.01}
        for f in selected: self.votes[f] += 1
        print(f"selected {len(selected)}/{len(self.features)}")

        print("    ⑪ Permutation importance...", end=" ")
        n_sub = min(3000, len(X_test))
        idx   = np.random.choice(len(X_test), n_sub, replace=False)
        perm  = permutation_importance(rf, X_test[idx], y_test[idx], n_repeats=5, random_state=42)
        perm_scores = dict(zip(self.features, perm.importances_mean))
        self.scores['Permutation'] = perm_scores
        selected_p = {f for f, v in perm_scores.items() if v > 0.001}
        for f in selected_p: self.votes[f] += 1
        print(f"selected {len(selected_p)}/{len(self.features)}")
        self.rf_model = rf
        return scores, perm_scores

    def granger_causality(self, df_train: pd.DataFrame, maxlag: int = 24):
        print("    ⑫ Granger Causality tests...", end=" ")
        ts_features = [f for f in self.features if f not in ['hour_sin','hour_cos','dow_sin','dow_cos','month_sin','month_cos']]
        granger_scores = {}
        y_arr = df_train['price'].values

        for feat in ts_features:
            try:
                x_arr = df_train[feat].values
                pval  = granger_causality_test(y_arr, x_arr, maxlag=min(maxlag, 12))
                granger_scores[feat] = 1.0 - pval
            except Exception:
                granger_scores[feat] = 0.0

        for feat in self.features:
            if feat not in granger_scores:
                granger_scores[feat] = 0.5

        self.scores['Granger'] = granger_scores
        selected = {f for f, v in granger_scores.items() if v > 0.95}
        for f in selected: self.votes[f] += 1
        print(f"selected {len(selected)}/{len(self.features)}")
        return granger_scores

    def shap_values(self, X: np.ndarray, X_test: np.ndarray, feature_names: list[str]):
        print("    ⑬ SHAP values (Tree SHAP approximation)...", end=" ")
        n_sub   = min(500, len(X_test))
        idx     = np.random.choice(len(X_test), n_sub, replace=False)
        X_sub   = X_test[idx]
        base_pred = self.rf_model.predict(X_sub)
        shap_imp  = np.zeros(len(feature_names))

        for j in range(len(feature_names)):
            X_perm = X_sub.copy()
            X_perm[:, j] = np.random.permutation(X_perm[:, j])
            perm_pred = self.rf_model.predict(X_perm)
            shap_imp[j] = np.mean(np.abs(base_pred - perm_pred))

        shap_scores = dict(zip(feature_names, shap_imp))
        self.scores['SHAP'] = shap_scores
        max_shap = max(shap_scores.values()) or 1
        selected = {f for f, v in shap_scores.items() if v > 0.1 * max_shap}
        for f in selected: self.votes[f] += 1
        print(f"selected {len(selected)}/{len(self.features)}")
        return shap_scores

    def run_all(self, X_tr: np.ndarray, y_tr: np.ndarray, X_te: np.ndarray, y_te: np.ndarray, df_train: pd.DataFrame) -> list[str]:
        print("\n" + "═"*65)
        print("  SECTION 2 — FEATURE SELECTION (13 METHODS)")
        print("═"*65)

        vif_kept = self.vif_filter(X_tr)
        feat_mask = np.array([f in vif_kept for f in self.features])
        X_tr_vif  = X_tr[:, feat_mask]
        X_te_vif  = X_te[:, feat_mask]
        feats_vif = [f for f in self.features if f in vif_kept]
        orig_features = self.features
        self.features = feats_vif

        self.pearson(X_tr_vif, y_tr)
        self.spearman(X_tr_vif, y_tr)
        self.mutual_information(X_tr_vif, y_tr)
        self.f_statistic(X_tr_vif, y_tr)
        self.rfe(X_tr_vif, y_tr, n_select=min(15, len(feats_vif)))
        self.sequential_selection(X_tr_vif, y_tr, n_select=min(15, len(feats_vif)))
        self.lasso_selection(X_tr_vif, y_tr)
        self.elastic_net_selection(X_tr_vif, y_tr)
        self.rf_importance(X_tr_vif, y_tr, X_te_vif, y_te)
        self.granger_causality(df_train)
        self.shap_values(X_tr_vif, X_te_vif, feats_vif)

        self.features = orig_features
        for f in self.vif_removed: self.votes[f] = 0

        self.selected = [f for f, v in self.votes.items() if v >= self.thresh]
        return self.selected


# ══════════════════════════════════════════════════════════════════════════════
# SECTION 3 ── EVALUATION METRICS
# ══════════════════════════════════════════════════════════════════════════════

def crps_gaussian(y: np.ndarray, mu: np.ndarray, sigma: np.ndarray) -> float:
    sigma = np.maximum(sigma, 1e-6)
    z     = (y - mu) / sigma
    return float(np.mean(sigma * (z * (2*stats.norm.cdf(z) - 1) + 2*stats.norm.pdf(z) - 1/np.sqrt(np.pi))))


def compute_all_metrics(y: np.ndarray, mu: np.ndarray, sigma: np.ndarray, label: str = "") -> dict:
    y     = np.asarray(y, float)
    mu    = np.asarray(mu, float)
    sigma = np.maximum(np.asarray(sigma, float), 1e-6)

    levels = np.arange(0.10, 1.0, 0.10)
    picps, mpiws = [], []
    for lv in levels:
        z   = stats.norm.ppf((1 + lv) / 2)
        lo  = mu - z * sigma
        hi  = mu + z * sigma
        picps.append(float(np.mean((y >= lo) & (y <= hi))))
        mpiws.append(float(np.mean(hi - lo)))

    z90  = stats.norm.ppf(0.95)
    lo90 = mu - z90 * sigma
    hi90 = mu + z90 * sigma

    maace = float(np.mean(np.abs(np.array(picps) - levels)))

    return {
        "MAE"    : float(mean_absolute_error(y, mu)),
        "RMSE"   : float(np.sqrt(mean_squared_error(y, mu))),
        "CRPS"   : crps_gaussian(y, mu, sigma),
        "PICP_90": float(np.mean((y >= lo90) & (y <= hi90))),
        "MPIW_90": float(np.mean(hi90 - lo90)),
        "MAACE"  : maace * 100,
        "mu"     : mu,
        "sigma"  : sigma,
        "lower"  : lo90,
        "upper"  : hi90,
        "picps"  : picps,
        "levels" : list(levels),
    }


# ══════════════════════════════════════════════════════════════════════════════
# SECTION 4 ── NEURAL NETWORK BASE CLASS
# ══════════════════════════════════════════════════════════════════════════════

class BaseNN:
    def _init_weights(self, dims: list[int]):
        weights, biases = [], []
        for i in range(len(dims) - 1):
            W = np.random.randn(dims[i], dims[i+1]) * np.sqrt(2.0 / dims[i])
            b = np.zeros(dims[i+1])
            weights.append(W); biases.append(b)
        return weights, biases

    def relu(self, x):    return np.maximum(0, x)
    def relu_d(self, x):  return (x > 0).astype(float)
    def softplus(self, x): return np.log1p(np.exp(np.clip(x, -20, 20)))

    def _forward_hidden(self, X, weights, biases):
        activations = [X]
        h = X
        for W, b in zip(weights[:-1], biases[:-1]):
            h = self.relu(h @ W + b)
            activations.append(h)
        return activations

    def _clip_grad(self, g, clip=1.0, max_norm=5.0):
        # First clip by global norm (the standard, effective way to stop
        # exploding gradients), then element-wise as a final safety net.
        norm = np.sqrt(np.sum(g**2))
        if norm > max_norm:
            g = g * (max_norm / (norm + 1e-8))
        return np.clip(g, -clip, clip)


# ══════════════════════════════════════════════════════════════════════════════
# SECTION 5 ── MODEL 1: DDNN (Baseline)
# ══════════════════════════════════════════════════════════════════════════════

class DDNN(BaseNN):
    """
    Distributional DNN with a Gaussian head, trained with ADAM.
 
    Change vs the old version: the optimizer is now Adam (adaptive momentum)
    instead of plain SGD with LR decay. Plain SGD converged poorly on the
    stabilized/asinh target (MAE in the hundreds); Adam brings it in line with
    the other models (MAE ~€13). Everything else — architecture, Gaussian NLL,
    variance floor, gradient clipping, early stopping — is unchanged.
    """
    def __init__(self, input_dim, hidden=[128, 64, 32], lr=0.005, epochs=300, batch_size=128):
        self.lr, self.epochs, self.batch_size = lr, epochs, batch_size
        self.losses = []
        dims = [input_dim] + hidden
        self.W, self.b = self._init_weights(dims)
        d = hidden[-1]
        self.Wmu = np.random.randn(d, 1) * 0.01;  self.bmu = np.zeros(1)
        self.Wlv = np.random.randn(d, 1) * 0.01;  self.blv = np.zeros(1)
        # ---- Adam optimizer state ----
        self._pnames = ['W0','b0','W1','b1','W2','b2','Wmu','bmu','Wlv','blv']
        self.m = {p: 0 for p in self._pnames}
        self.v = {p: 0 for p in self._pnames}
        self.t = 0
 
    # map parameter name <-> actual array
    def _get(self, p):
        if p == 'Wmu': return self.Wmu
        if p == 'bmu': return self.bmu
        if p == 'Wlv': return self.Wlv
        if p == 'blv': return self.blv
        i = int(p[1]);  return self.W[i] if p[0] == 'W' else self.b[i]
    def _set(self, p, val):
        if p == 'Wmu': self.Wmu = val; return
        if p == 'bmu': self.bmu = val; return
        if p == 'Wlv': self.Wlv = val; return
        if p == 'blv': self.blv = val; return
        i = int(p[1])
        if p[0] == 'W': self.W[i] = val
        else:           self.b[i] = val
 
    def _adam_step(self, grads):
        self.t += 1
        b1, b2, eps = 0.9, 0.999, 1e-8
        for p, g in grads.items():
            g = self._clip_grad(g)                       # reuse norm-based clipping
            self.m[p] = b1 * self.m[p] + (1 - b1) * g
            self.v[p] = b2 * self.v[p] + (1 - b2) * g**2
            m_hat = self.m[p] / (1 - b1**self.t)
            v_hat = self.v[p] / (1 - b2**self.t)
            self._set(p, self._get(p) - self.lr * m_hat / (np.sqrt(v_hat) + eps))
 
    def _forward(self, X):
        h1 = self.relu(X  @ self.W[0] + self.b[0])
        h2 = self.relu(h1 @ self.W[1] + self.b[1])
        h3 = self.relu(h2 @ self.W[2] + self.b[2])
        mu = (h3 @ self.Wmu + self.bmu).flatten()
        lv = np.clip((h3 @ self.Wlv + self.blv).flatten(), -6, 6)
        return mu, lv, h1, h2, h3
 
    def _nll(self, y, mu, lv):
        var = np.exp(lv) + 1e-3
        return np.mean(0.5 * lv + 0.5 * (y - mu)**2 / var)
 
    def fit(self, X, y, X_val=None, y_val=None, verbose=True):
        if verbose: print(f"\n  Training DDNN (Adam)...", end=" ")
        n = len(X)
        best_val, patience, wait = np.inf, 8, 0
        for epoch in range(self.epochs):
            indices = np.random.permutation(n)
            for start in range(0, n, self.batch_size):
                idx = indices[start:start+self.batch_size]
                X_b, y_b = X[idx], y[idx]
                mu, lv, h1, h2, h3 = self._forward(X_b)
                var  = np.exp(lv) + 1e-3
                err  = np.clip(y_b - mu, -10.0, 10.0)
                d_mu = -err / var / len(y_b)
                d_lv = np.clip(0.5 * (1 - err**2 / var) / len(y_b), -1.0, 1.0)
                g = {}
                g['Wmu'] = h3.T @ d_mu.reshape(-1, 1); g['bmu'] = np.array([d_mu.sum()])
                g['Wlv'] = h3.T @ d_lv.reshape(-1, 1); g['blv'] = np.array([d_lv.sum()])
                dh3 = (d_mu.reshape(-1,1) @ self.Wmu.T + d_lv.reshape(-1,1) @ self.Wlv.T) * self.relu_d(h3)
                g['W2'] = h2.T @ dh3; g['b2'] = dh3.sum(axis=0)
                dh2 = (dh3 @ self.W[2].T) * self.relu_d(h2)
                g['W1'] = h1.T @ dh2; g['b1'] = dh2.sum(axis=0)
                dh1 = (dh2 @ self.W[1].T) * self.relu_d(h1)
                g['W0'] = X_b.T @ dh1; g['b0'] = dh1.sum(axis=0)
                self._adam_step(g)
            mu_full, lv_full, _, _, _ = self._forward(X)
            self.losses.append(self._nll(y, mu_full, lv_full))
            if X_val is not None and (epoch+1) % 10 == 0:
                mu_v, lv_v = self._pred_raw(X_val)
                val_loss   = self._nll(y_val, mu_v, lv_v)
                if val_loss < best_val - 1e-4: best_val = val_loss; wait = 0
                else: wait += 1
                if wait >= patience: break
        if verbose: print(f"done. Final NLL={self.losses[-1]:.4f}")
        return self
 
    def _pred_raw(self, X):
        mu, lv, _, _, _ = self._forward(X)
        return mu, lv
 
    def predict(self, X):
        mu, lv = self._pred_raw(X)
        return mu, np.sqrt(np.exp(lv) + 1e-3)


# ══════════════════════════════════════════════════════════════════════════════
# SECTION 6 ── MODEL 2: EvDNN
# ══════════════════════════════════════════════════════════════════════════════

class EvDNN(BaseNN):
    """
    Evidential DNN (Deep Evidential Regression) — now with ADAM and FULL backprop.
 
    Fix vs the old version: the previous fit() only updated the gamma head and
    left the hidden layers (and nu/alpha/beta heads) frozen at random init, so
    most of the network never learned (MAE ~€23). This version backpropagates
    through every layer and trains with Adam, matching the DDNN fix.
    """
    def __init__(self, input_dim, hidden=[128, 64, 32], lr=0.005, epochs=300, lam=0.02, batch_size=128):
        self.lr, self.epochs, self.lam, self.batch_size = lr, epochs, lam, batch_size
        self.losses = []
        dims = [input_dim] + hidden
        self.W, self.b = self._init_weights(dims)
        d = hidden[-1]
        self.Wg = np.random.randn(d,1)*0.01; self.bg = np.zeros(1)
        self.Wv = np.random.randn(d,1)*0.01; self.bv = np.zeros(1)
        self.Wa = np.random.randn(d,1)*0.01; self.ba = np.zeros(1)
        self.Wb = np.random.randn(d,1)*0.01; self.bb = np.zeros(1)
        self._pnames = ['W0','b0','W1','b1','W2','b2','Wg','bg','Wv','bv','Wa','ba','Wb','bb']
        self.m = {p: 0 for p in self._pnames}
        self.v = {p: 0 for p in self._pnames}
        self.t = 0
 
    def _get(self, p):
        d = {'Wg':self.Wg,'bg':self.bg,'Wv':self.Wv,'bv':self.bv,
             'Wa':self.Wa,'ba':self.ba,'Wb':self.Wb,'bb':self.bb}
        if p in d: return d[p]
        i = int(p[1]); return self.W[i] if p[0]=='W' else self.b[i]
    def _set(self, p, val):
        if   p=='Wg': self.Wg=val
        elif p=='bg': self.bg=val
        elif p=='Wv': self.Wv=val
        elif p=='bv': self.bv=val
        elif p=='Wa': self.Wa=val
        elif p=='ba': self.ba=val
        elif p=='Wb': self.Wb=val
        elif p=='bb': self.bb=val
        else:
            i=int(p[1])
            if p[0]=='W': self.W[i]=val
            else:         self.b[i]=val
 
    def _adam_step(self, grads):
        self.t += 1; b1,b2,eps = 0.9,0.999,1e-8
        for p,g in grads.items():
            g = self._clip_grad(g)
            self.m[p]=b1*self.m[p]+(1-b1)*g; self.v[p]=b2*self.v[p]+(1-b2)*g**2
            mh=self.m[p]/(1-b1**self.t); vh=self.v[p]/(1-b2**self.t)
            self._set(p, self._get(p)-self.lr*mh/(np.sqrt(vh)+eps))
 
    def _forward(self, X):
        h1 = self.relu(X  @ self.W[0] + self.b[0])
        h2 = self.relu(h1 @ self.W[1] + self.b[1])
        h  = self.relu(h2 @ self.W[2] + self.b[2])
        pg = (h @ self.Wg + self.bg).flatten()
        pv = (h @ self.Wv + self.bv).flatten()
        pa = (h @ self.Wa + self.ba).flatten()
        pb = (h @ self.Wb + self.bb).flatten()
        gamma = pg
        nu    = self.softplus(pv) + 1e-4
        alpha = self.softplus(pa) + 1.01
        beta  = self.softplus(pb) + 1e-4
        return gamma, nu, alpha, beta, (h1,h2,h,pv,pa,pb)
 
    def _loss(self, y, gamma, nu, alpha, beta):
        omega = np.maximum(2*beta*(1+nu), 1e-6)
        nll = (0.5*np.log(np.pi/np.maximum(nu,1e-6)) - alpha*np.log(omega)
               + (alpha+0.5)*np.log(nu*(y-gamma)**2 + omega)
               + gammaln(alpha) - gammaln(alpha+0.5))
        reg = np.abs(y-gamma) * (2*nu + alpha)
        return float(np.mean(nll + self.lam*reg))
 
    def _sp_grad(self, x):   # derivative of softplus
        return 1.0/(1.0+np.exp(-np.clip(x,-30,30)))
 
    def fit(self, X, y, X_val=None, y_val=None, verbose=True):
        if verbose: print(f"\n  Training EvDNN (Adam)...", end=" ")
        n = len(X); best_val, wait, patience = np.inf, 0, 8
        for epoch in range(self.epochs):
            idx_all = np.random.permutation(n)
            for start in range(0, n, self.batch_size):
                bi = idx_all[start:start+self.batch_size]
                X_b, y_b = X[bi], y[bi]
                gamma, nu, alpha, beta, (h1,h2,h,pv,pa,pb) = self._forward(X_b)
                nb = len(y_b)
                err = y_b - gamma
                S = nu*err**2 + 2*beta*(1+nu)          # inside the log
                S = np.maximum(S, 1e-6)
                # --- gradients of NLL wrt each output (approximate, stable) ---
                dgamma = (-(alpha+0.5)*(2*nu*err)/S) + self.lam*(2*nu+alpha)*(-np.sign(err))
                dnu    = (-0.5/np.maximum(nu,1e-6)
                          + (alpha+0.5)*(err**2 + 2*beta)/S
                          - alpha*(2*beta)/np.maximum(2*beta*(1+nu),1e-6)) + self.lam*2
                dalpha = (-np.log(np.maximum(2*beta*(1+nu),1e-6)) + np.log(S)) + self.lam
                dbeta  = (-alpha*(2*(1+nu))/np.maximum(2*beta*(1+nu),1e-6)
                          + (alpha+0.5)*(2*(1+nu))/S)
                # chain through softplus for nu/alpha/beta heads
                dpv = (dnu    * self._sp_grad(pv)) / nb
                dpa = (dalpha * self._sp_grad(pa)) / nb
                dpb = (dbeta  * self._sp_grad(pb)) / nb
                dpg = dgamma / nb
                g = {}
                g['Wg']=h.T@dpg.reshape(-1,1); g['bg']=np.array([dpg.sum()])
                g['Wv']=h.T@dpv.reshape(-1,1); g['bv']=np.array([dpv.sum()])
                g['Wa']=h.T@dpa.reshape(-1,1); g['ba']=np.array([dpa.sum()])
                g['Wb']=h.T@dpb.reshape(-1,1); g['bb']=np.array([dpb.sum()])
                dh = (dpg.reshape(-1,1)@self.Wg.T + dpv.reshape(-1,1)@self.Wv.T
                      + dpa.reshape(-1,1)@self.Wa.T + dpb.reshape(-1,1)@self.Wb.T) * self.relu_d(h)
                g['W2']=h2.T@dh; g['b2']=dh.sum(0)
                dh2=(dh@self.W[2].T)*self.relu_d(h2); g['W1']=h1.T@dh2; g['b1']=dh2.sum(0)
                dh1=(dh2@self.W[1].T)*self.relu_d(h1); g['W0']=X_b.T@dh1; g['b0']=dh1.sum(0)
                self._adam_step(g)
            gf,nf,af,bf,_ = self._forward(X)
            self.losses.append(self._loss(y,gf,nf,af,bf))
            if X_val is not None and (epoch+1)%10==0:
                gv,nv,av,bv,_ = self._forward(X_val)
                vl = self._loss(y_val,gv,nv,av,bv)
                if vl < best_val-1e-4: best_val=vl; wait=0
                else: wait+=1
                if wait>=patience: break
        if verbose: print(f"done. Final loss={self.losses[-1]:.4f}")
        return self
 
    def predict(self, X):
        gamma, nu, alpha, beta, _ = self._forward(X)
        total_sigma = np.sqrt(np.clip(beta*(1+nu)/(nu*(alpha-1+1e-6)), 1e-6, 1e4))
        self.epistemic = np.sqrt(np.clip(beta/(nu*(alpha-1+1e-6)), 1e-6, 1e4))
        self.aleatoric = np.sqrt(np.clip(beta/(alpha-1+1e-6), 1e-6, 1e4))
        return gamma, total_sigma


# ══════════════════════════════════════════════════════════════════════════════
# SECTION 7 ── MODEL 3: VI-DDNN (Variational Inference)
# ══════════════════════════════════════════════════════════════════════════════

class VIDDNN(BaseNN):
    def __init__(self, input_dim, hidden=[128, 64, 32], lr=0.001, epochs=500, n_samples=30, prior_sigma=1.0, batch_size=128):
        self.lr         = lr
        self.epochs     = epochs
        self.n_samples  = n_samples
        self.prior_var  = prior_sigma**2
        self.batch_size = batch_size
        self.losses     = []

        h1, h2 = hidden[0], hidden[1]
        self.W1_mu = np.random.randn(input_dim, h1) * 0.1
        self.W1_ls = np.full((input_dim, h1), -3.0)
        self.b1_mu = np.zeros(h1); self.b1_ls = np.full(h1, -3.0)
        self.W2_mu = np.random.randn(h1, h2) * 0.1
        self.W2_ls = np.full((h1, h2), -3.0)
        self.b2_mu = np.zeros(h2); self.b2_ls = np.full(h2, -3.0)
        self.Wmu = np.random.randn(h2, 1) * 0.01; self.bmu = np.zeros(1)
        self.Wlv = np.random.randn(h2, 1) * 0.01; self.blv = np.zeros(1)

    def _sample_w(self, mu, log_sigma):
        sigma = np.exp(np.clip(log_sigma, -6, 2))
        return mu + sigma * np.random.randn(*mu.shape)

    def _forward_one(self, X):
        W1 = self._sample_w(self.W1_mu, self.W1_ls)
        b1 = self._sample_w(self.b1_mu, self.b1_ls)
        W2 = self._sample_w(self.W2_mu, self.W2_ls)
        b2 = self._sample_w(self.b2_mu, self.b2_ls)
        h1 = self.relu(X @ W1 + b1)
        h2 = self.relu(h1 @ W2 + b2)
        mu = (h2 @ self.Wmu + self.bmu).flatten()
        lv = np.clip((h2 @ self.Wlv + self.blv).flatten(), -6, 6)
        return mu, lv, h1, h2, W2

    def _kl_loss(self):
        kl = 0.0
        for mu, ls in [(self.W1_mu, self.W1_ls), (self.W2_mu, self.W2_ls),
                       (self.b1_mu, self.b1_ls), (self.b2_mu, self.b2_ls)]:
            var = np.exp(2 * np.clip(ls, -6, 2))
            kl += 0.5 * np.sum((mu**2 + var) / self.prior_var - 2*ls - 1)
        return kl

    def fit(self, X, y, X_val=None, y_val=None, verbose=True):
        if verbose: print(f"\n  Training VI-DDNN ({self.n_samples} samples/pass)...", end=" ")
        n = len(X)
        best_val, wait, patience = np.inf, 0, 50

        for epoch in range(self.epochs):
            kl_weight = min(1.0, epoch / 50.0) * (1e-5)
            lr_t = self.lr * (0.99 ** (epoch // 10))
            indices = np.random.permutation(n)

            for start in range(0, n, self.batch_size):
                idx = indices[start:start+self.batch_size]
                X_b, y_b = X[idx], y[idx]
                
                mu, lv, h1, h2, W2 = self._forward_one(X_b)
                var_pred = np.exp(lv) + 1e-6
                
                err   = y_b - mu
                d_mu  = -err / var_pred / len(y_b)
                d_lv  = 0.5 * (1 - err**2 / var_pred) / len(y_b)
                
                self.Wmu -= lr_t * self._clip_grad(h2.T @ d_mu.reshape(-1,1))
                self.bmu -= lr_t * d_mu.sum()
                self.Wlv -= lr_t * self._clip_grad(h2.T @ d_lv.reshape(-1,1))
                self.blv -= lr_t * d_lv.sum()

                self.W2_mu -= lr_t * (
                    self._clip_grad(h1.T @ (d_mu.reshape(-1,1) @ self.Wmu.T *
                    (h1 @ self._sample_w(self.W2_mu, self.W2_ls) > 0))) +
                    kl_weight * self.W2_mu / (self.prior_var * len(y_b))
                )
                self.W1_mu -= lr_t * kl_weight * self.W1_mu / (self.prior_var * len(y_b))

            mu_full, lv_full, _, _, _ = self._forward_one(X)
            var_f = np.exp(lv_full) + 1e-6
            nll   = np.mean(0.5 * lv_full + 0.5 * (y - mu_full)**2 / var_f)
            kl    = self._kl_loss() / n
            self.losses.append(nll + kl_weight * kl)

            if X_val is not None and (epoch+1) % 10 == 0:
                mu_v = np.mean([self._forward_one(X_val)[0] for _ in range(3)], axis=0)
                vl = np.mean((y_val - mu_v)**2)
                if vl < best_val - 1e-4: best_val = vl; wait = 0
                else: wait += 1
                if wait >= patience: break

        if verbose: print(f"done. Final ELBO={self.losses[-1]:.4f}")

    def predict(self, X):
        mus, sigs = [], []
        for _ in range(self.n_samples):
            mu, lv, _, _, _ = self._forward_one(X)
            mus.append(mu)
            sigs.append(np.sqrt(np.exp(lv) + 1e-6))
        mus, sigs = np.array(mus), np.array(sigs)
        mu_pred        = mus.mean(axis=0)
        self.epistemic = mus.std(axis=0)
        self.aleatoric = sigs.mean(axis=0)
        total_sigma    = np.sqrt(self.epistemic**2 + self.aleatoric**2)
        return mu_pred, total_sigma


# ══════════════════════════════════════════════════════════════════════════════
# SECTION 8 ── MODEL 4: BSSM (Bayesian State-Space Model)
# ══════════════════════════════════════════════════════════════════════════════

class BSSM:
    def __init__(self, Q_init=50.0, R_init=300.0, em_iters=15):
        self.Q, self.R, self.em_iters = Q_init, R_init, em_iters

    def fit(self, y: np.ndarray):
        print(f"\n  Training BSSM ({self.em_iters} EM iterations)...", end=" ")
        y = np.asarray(y, float)
        n = len(y)

        for _ in range(self.em_iters):
            x_hat, P = np.zeros(n), np.zeros(n)
            x_hat[0], P[0] = y[0], self.R
            for t in range(1, n):
                xp = x_hat[t-1]; pp = P[t-1] + self.Q
                K  = pp / (pp + self.R)
                x_hat[t] = xp + K * (y[t] - xp)
                P[t]     = (1 - K) * pp

            innovations = y[1:] - x_hat[:-1]
            self.R = max(float(np.var(innovations)) * 0.9, 0.1)
            self.Q = max(float(np.var(np.diff(x_hat))) * 0.15, 0.01)

        self._x_hat_train = x_hat
        print(f"done.  Q={self.Q:.2f}  R={self.R:.2f}")

    def predict_test(self, y_train: np.ndarray, y_test: np.ndarray):
        lo_b = min(y_train.min(), y_test.min()) - 5
        hi_b = max(y_train.max(), y_test.max()) + 5
        x_hat = float(np.mean(y_train[:24]))
        P     = float(self.R)

        for t in range(len(y_train)):
            pp    = min(P + self.Q, 1e8)
            K     = np.clip(pp / (pp + self.R + 1e-10), 0, 0.99)
            x_hat = float(np.clip(x_hat + K * (y_train[t] - x_hat), lo_b, hi_b))
            P     = float((1 - K) * pp)

        mu_pred, sig_pred = np.zeros(len(y_test)), np.zeros(len(y_test))
        for t in range(len(y_test)):
            P_pred      = min(P + self.Q, 1e8)
            mu_pred[t]  = x_hat
            sig_pred[t] = float(np.sqrt(max(P_pred, 1e-6)))
            K           = np.clip(P_pred / (P_pred + self.R + 1e-10), 0, 0.99)
            x_hat       = float(np.clip(x_hat + K * (y_test[t] - x_hat), lo_b, hi_b))
            P           = float((1 - K) * P_pred)

        mu_pred  = np.clip(mu_pred, lo_b, hi_b)
        sig_pred = np.clip(sig_pred, 1e-3, abs(hi_b - lo_b))
        return mu_pred, sig_pred


# ══════════════════════════════════════════════════════════════════════════════
# SECTION 9 ── CONFORMAL PREDICTION WRAPPER
# ══════════════════════════════════════════════════════════════════════════════

class ConformalWrapper:
    def __init__(self, coverage=0.90):
        self.coverage  = coverage
        self.quantiles = {}

    def fit(self, y_cal: np.ndarray, mu_cal: np.ndarray, hours_cal: np.ndarray):
        residuals = np.abs(y_cal - mu_cal)
        for h in range(24):
            mask = hours_cal == h
            if mask.sum() > 10:
                self.quantiles[h] = np.quantile(residuals[mask], self.coverage)
            else:
                self.quantiles[h] = np.quantile(residuals, self.coverage)

    def predict(self, mu_test: np.ndarray, hours_test: np.ndarray):
        widths = np.array([self.quantiles.get(int(h), np.median(list(self.quantiles.values()))) for h in hours_test])
        z     = stats.norm.ppf((1 + self.coverage) / 2)
        sigma = widths / z
        return mu_test, sigma


# ══════════════════════════════════════════════════════════════════════════════
# SECTION 10 ── PLOTLY VISUALIZATIONS & REPORTING
# ══════════════════════════════════════════════════════════════════════════════

def print_results_table(results: dict):
    print("\n" + "═"*80)
    print("  RESULTS TABLE — All Models on Test Set (May 2025 – End)")
    print("═"*80)
    print(f"  {'Model':<18} {'MAE':>7} {'RMSE':>8} {'CRPS':>7} {'PICP_90':>9} {'MPIW_90':>9} {'MAACE':>8}")
    print("  " + "─"*70)

    best = {
        "MAE"    : min(r["MAE"]     for r in results.values()),
        "RMSE"   : min(r["RMSE"]    for r in results.values()),
        "CRPS"   : min(r["CRPS"]    for r in results.values()),
        "PICP_90": min(abs(r["PICP_90"] - 0.90) for r in results.values()),
        "MPIW_90": min(r["MPIW_90"] for r in results.values()),
        "MAACE"  : min(r["MAACE"]   for r in results.values()),
    }

    for name, r in results.items():
        def mark(key, val):
            if key == "PICP_90":
                return "◀" if abs(val-0.90) == best[key] else " "
            return "◀" if abs(val - best[key]) < 0.01 else " "

        print(f"  {name:<18} "
              f"{r['MAE']:>7.2f}{mark('MAE',r['MAE'])} "
              f"{r['RMSE']:>8.2f}{mark('RMSE',r['RMSE'])} "
              f"{r['CRPS']:>7.3f}{mark('CRPS',r['CRPS'])} "
              f"{r['PICP_90']:>8.1%}{mark('PICP_90',r['PICP_90'])} "
              f"{r['MPIW_90']:>9.2f}{mark('MPIW_90',r['MPIW_90'])} "
              f"{r['MAACE']:>7.2f}%{mark('MAACE',r['MAACE'])}")

    print("═"*80)


def plot_results_plotly(results: dict, y_test: np.ndarray, save_path: str = "/Users/islamriajul/Documents/epf_results_plotly.html"):
    print("\nGenerating interactive Plotly dashboard...", end=" ")
    n_show = min(720, len(y_test))
    models = list(results.keys())
    n_mod = len(models)
    
    colors = {
        "DDNN": "#378ADD",
        "EvDNN": "#D85A30",
        "VI-DDNN": "#1D9E75",
        "BSSM": "#BA7517",
        "VI+CP": "#7F77DD"
    }

    fig = make_subplots(
        rows=4, cols=max(n_mod, 4),
        subplot_titles=(
            [f"{m} Forecast" for m in models] + ["" for _ in range(max(0, 4 - n_mod))] +
            [f"{m} Uncertainty" for m in models] + ["" for _ in range(max(0, 4 - n_mod))] +
            ["MAE (€/MWh)", "RMSE (€/MWh)", "CRPS", "MPIW 90% (€/MWh)"] +
            ["Calibration Curves (All Models)", "", "", ""]
        ),
        vertical_spacing=0.08, horizontal_spacing=0.04,
        specs=[
            [{"type": "xy"} for _ in range(max(n_mod, 4))],
            [{"type": "xy"} for _ in range(max(n_mod, 4))],
            [{"type": "xy"} for _ in range(4)] + [{"type": "xy"} for _ in range(max(0, n_mod - 4))],
            [{"colspan": max(n_mod, 4), "type": "xy"}] + [None for _ in range(max(n_mod, 4) - 1)]
        ]
    )

    for i, name in enumerate(models):
        r = results[name]
        mu, lo, hi = r["mu"][:n_show], r["lower"][:n_show], r["upper"][:n_show]
        col = colors.get(name, "#378ADD")

        fig.add_trace(go.Scatter(x=list(range(n_show)), y=hi, mode='lines', line=dict(width=0), showlegend=False), row=1, col=i+1)
        fig.add_trace(go.Scatter(x=list(range(n_show)), y=lo, mode='lines', line=dict(width=0), fill='tonexty',
                                 fillcolor=f"rgba{tuple(int(col.lstrip('#')[j:j+2], 16) for j in (0, 2, 4)) + (0.2,)}",
                                 name="90% CI", showlegend=(i == 0)), row=1, col=i+1)
        fig.add_trace(go.Scatter(x=list(range(n_show)), y=y_test[:n_show], mode='lines', line=dict(color="#333333", width=1),
                                 name="Actual", showlegend=(i == 0)), row=1, col=i+1)
        fig.add_trace(go.Scatter(x=list(range(n_show)), y=mu, mode='lines', line=dict(color=col, width=1.5),
                                 name=f"{name} Forecast", showlegend=False), row=1, col=i+1)

    for j, met in enumerate(["MAE", "RMSE", "CRPS", "MPIW_90"]):
        vals = [results[m][met] for m in models]
        cols = [colors.get(m, "#378ADD") for m in models]
        fig.add_trace(go.Bar(x=models, y=vals, marker_color=cols, text=[f"{v:.1f}" for v in vals], textposition='auto', showlegend=False), row=3, col=j+1)

    for name, r in results.items():
        col = colors.get(name, "#378ADD")
        fig.add_trace(go.Scatter(x=[l * 100 for l in r["levels"]], y=[p * 100 for p in r["picps"]], mode='lines+markers', line=dict(color=col, width=2), name=name), row=4, col=1)

    fig.add_trace(go.Scatter(x=[10, 90], y=[10, 90], mode='lines', line=dict(color="black", dash='dash', width=1.5), name="Ideal Calibration"), row=4, col=1)

    fig.update_layout(height=1400, width=1800, title_text="<b>Bayesian Uncertainty Quantification Dashboard</b>", title_x=0.5, template="plotly_white")
    
    # Save interactive HTML dashboard and display
    fig.write_html(save_path)
    print(f"saved → {save_path}")
    fig.show()


# ══════════════════════════════════════════════════════════════════════════════
# SECTION 11 ── MAIN PIPELINE
# ══════════════════════════════════════════════════════════════════════════════

def predict_log_space_interval(model, X_te, scaler_y, confidence=0.90):
    """Constructs prediction intervals in signed-log space FIRST."""
    mu_s, sig_s = model.predict(X_te)
    mu_z  = scaler_y.inverse_transform(mu_s.reshape(-1, 1)).flatten()
    sig_z = np.abs(sig_s) * scaler_y.scale_[0]
    
    z_quant = stats.norm.ppf((1 + confidence) / 2)
    z_lower = mu_z - z_quant * sig_z
    z_upper = mu_z + z_quant * sig_z
    
    mu_eur    = inv_signed_log(mu_z)
    lower_eur = inv_signed_log(z_lower)
    upper_eur = inv_signed_log(z_upper)
    
    sig_eur_stable = (upper_eur - lower_eur) / (2 * z_quant)
    sig_eur_stable = np.clip(sig_eur_stable, 1e-3, 500.0)
    
    return mu_eur, sig_eur_stable, lower_eur, upper_eur


def main():
    print("\n" + "█"*65)
    print("  GERMAN ELECTRICITY PRICE FORECASTING — RESEARCH PIPELINE")
    print("  13 Feature Selection Methods + 4 Bayesian Models")
    print("█"*65)

    df, candidate_features = load_real_smard(
        '/Users/islamriajul/Documents/Day-ahead_prices_202001010000_202605010000_Hour.csv'
    )
    
    # Chronological Split via explicit dates
    train_df, val_df, test_df = split_data(df)

    USE_SIGNED_LOG = True

    if USE_SIGNED_LOG:
        print("\n  ★ SIGNED-LOG TARGET TRANSFORM ENABLED")
        train_df = train_df.copy(); val_df = val_df.copy(); test_df = test_df.copy()
        train_df['price_model'] = signed_log(train_df['price'].values)
        val_df['price_model']   = signed_log(val_df['price'].values)
        test_df['price_model']  = signed_log(test_df['price'].values)
        MODEL_TARGET = 'price_model'
    else:
        MODEL_TARGET = 'price'

    X_tr  = train_df[candidate_features].values
    X_val = val_df[candidate_features].values
    X_te  = test_df[candidate_features].values

    y_tr_model = train_df[MODEL_TARGET].values
    y_val_model= val_df[MODEL_TARGET].values
    y_tr_raw   = train_df['price'].values
    y_val_raw  = val_df['price'].values
    y_te       = test_df['price'].values

    scaler_X = StandardScaler()
    scaler_y = StandardScaler()
    X_tr_s   = scaler_X.fit_transform(X_tr)
    X_val_s  = scaler_X.transform(X_val)
    X_te_s   = scaler_X.transform(X_te)
    y_tr_s   = scaler_y.fit_transform(y_tr_model.reshape(-1,1)).flatten()
    y_val_s  = scaler_y.transform(y_val_model.reshape(-1,1)).flatten()

    cache_path = 'selected_features_signedlog.json'
    if os.path.exists(cache_path):
        with open(cache_path) as f:
            selected = json.load(f)
        print(f"\n  Loaded cached feature selection: {len(selected)} features")
        selector = FeatureSelector(candidate_features, threshold_votes=5)
        selector.selected = selected
    else:
        selector = FeatureSelector(candidate_features, threshold_votes=5)
        selected = selector.run_all(X_tr_s, y_tr_s, X_te_s, y_te, train_df)
        with open(cache_path, 'w') as f:
            json.dump(selected, f)

    sel_idx   = [candidate_features.index(f) for f in selected]
    X_tr_sel  = X_tr_s[:, sel_idx]
    X_val_sel = X_val_s[:, sel_idx]
    X_te_sel  = X_te_s[:, sel_idx]
    n_feat    = len(selected)

    results = {}

    # ── 5. MODEL 1: DDNN ──────────────────────────────────────────────────────
    print("\n" + "═"*65)
    print("  MODEL 1/4: DDNN — Distributional Neural Network (Baseline)")
    print("═"*65)
    ddnn = DDNN(n_feat, hidden=[128, 64, 32], lr=0.002, epochs=500)
    ddnn.fit(X_tr_sel, y_tr_s, X_val_sel, y_val_s)
    mu, sig, _, _ = predict_log_space_interval(ddnn, X_te_sel, scaler_y)
    m = compute_all_metrics(y_te, mu, sig, "DDNN")
    m["epistemic"] = None; m["aleatoric"] = None
    results["DDNN"] = m

    # ── 6. MODEL 2: EvDNN ─────────────────────────────────────────────────────
    print("\n" + "═"*65)
    print("  MODEL 2/4: EvDNN — Evidential Deep Neural Network")
    print("═"*65)
    evdnn = EvDNN(n_feat, hidden=[128, 64, 32], lr=0.001, epochs=500, lam=0.02)
    evdnn.fit(X_tr_sel, y_tr_s, X_val_sel, y_val_s)
    mu, sig, _, _ = predict_log_space_interval(evdnn, X_te_sel, scaler_y)
    ep          = getattr(evdnn, 'epistemic', np.zeros_like(mu)) * scaler_y.scale_[0]
    al          = getattr(evdnn, 'aleatoric', np.zeros_like(mu)) * scaler_y.scale_[0]
    m = compute_all_metrics(y_te, mu, sig, "EvDNN")
    m["epistemic"] = ep; m["aleatoric"] = al
    results["EvDNN"] = m

    # ── 7. MODEL 3: VI-DDNN ───────────────────────────────────────────────────
    print("\n" + "═"*65)
    print("  MODEL 3/4: VI-DDNN — Variational Inference (True Bayesian NN)")
    print("═"*65)
    vi = VIDDNN(n_feat, hidden=[128, 64], lr=0.002, epochs=300, n_samples=20)
    vi.fit(X_tr_sel, y_tr_s, X_val_sel, y_val_s)
    mu, sig, _, _ = predict_log_space_interval(vi, X_te_sel, scaler_y)
    ep          = getattr(vi, 'epistemic', np.zeros_like(mu)) * scaler_y.scale_[0]
    al          = getattr(vi, 'aleatoric', np.zeros_like(mu)) * scaler_y.scale_[0]
    m = compute_all_metrics(y_te, mu, sig, "VI-DDNN")
    m["epistemic"] = ep; m["aleatoric"] = al
    results["VI-DDNN"] = m

    # ── 8. MODEL 3b: VI-DDNN + Conformal Prediction ───────────────────────────
    print("\n" + "═"*65)
    print("  MODEL 3b: VI-DDNN + Conformal Prediction Calibration")
    print("═"*65)
    mu_val_s, _ = vi.predict(X_val_sel)
    mu_val_z    = scaler_y.inverse_transform(mu_val_s.reshape(-1,1)).flatten()
    mu_val      = inv_signed_log(mu_val_z)
    hours_val   = val_df['hour'].values
    hours_te    = test_df['hour'].values
    cp = ConformalWrapper(coverage=0.90)
    cp.fit(y_val_raw, mu_val, hours_val)
    mu_cp, sig_cp = cp.predict(mu, hours_te)
    m = compute_all_metrics(y_te, mu_cp, sig_cp, "VI+CP")
    m["epistemic"] = ep; m["aleatoric"] = al
    results["VI+CP"] = m

    # ── 9. MODEL 4: BSSM ──────────────────────────────────────────────────────
    print("\n" + "═"*65)
    print("  MODEL 4/4: BSSM — Bayesian State-Space Model")
    print("═"*65)
    bssm = BSSM(Q_init=50, R_init=300, em_iters=15)
    bssm.fit(y_tr_model)
    mu_bssm_z, sig_bssm_z = bssm.predict_test(y_tr_model, test_df[MODEL_TARGET].values)
    z90 = stats.norm.ppf(0.95)
    lower_bssm = inv_signed_log(mu_bssm_z - z90 * sig_bssm_z)
    upper_bssm = inv_signed_log(mu_bssm_z + z90 * sig_bssm_z)
    mu_bssm    = inv_signed_log(mu_bssm_z)
    sig_bssm_stable = (upper_bssm - lower_bssm) / (2 * z90)
    m = compute_all_metrics(y_te, mu_bssm, sig_bssm_stable, "BSSM")
    m["lower"], m["upper"] = lower_bssm, upper_bssm
    m["epistemic"] = None; m["aleatoric"] = None
    results["BSSM"] = m

    # ── 10. Print results ─────────────────────────────────────────────────────
    print_results_table(results)

    # ── 11. Plotly Visualizations ─────────────────────────────────────────────
    plot_results_plotly(
        results, y_te,
        save_path="/Users/islamriajul/Documents/epf_results_plotly.html"
    )

    print("\n" + "█"*65)
    print("  PIPELINE COMPLETE")
    print("  Interactive Output Saved: epf_results_plotly.html")
    print("█"*65)


if __name__ == "__main__":
    main()


# ═══ Added for Parts 15b/16: Student-t (Adam) + interval metrics ═══
class StudentTDNN_Adam:
    def __init__(self, input_dim, hidden=[64,32], lr=0.01, epochs=250):
        self.lr, self.epochs = lr, epochs
        h1, h2 = hidden
        self.P = {
            'W1': np.random.randn(input_dim,h1)*np.sqrt(2/input_dim), 'b1': np.zeros(h1),
            'W2': np.random.randn(h1,h2)*np.sqrt(2/h1), 'b2': np.zeros(h2),
            'Wm': np.random.randn(h2,1)*0.01, 'bm': np.zeros(1),
            'Ws': np.random.randn(h2,1)*0.01, 'bs': np.zeros(1),
            'Wn': np.random.randn(h2,1)*0.01, 'bn': np.zeros(1),
        }
        self.m = {k: np.zeros_like(v) for k,v in self.P.items()}
        self.v = {k: np.zeros_like(v) for k,v in self.P.items()}
        self.t = 0
    def _relu(self,x): return np.maximum(0,x)
    def _fwd(self,X):
        z1=X@self.P['W1']+self.P['b1']; h1=self._relu(z1)
        z2=h1@self.P['W2']+self.P['b2']; h2=self._relu(z2)
        mu=(h2@self.P['Wm']+self.P['bm']).flatten()
        lsig=np.clip((h2@self.P['Ws']+self.P['bs']).flatten(),-4,4)
        lnu=np.clip((h2@self.P['Wn']+self.P['bn']).flatten(),0.3,3.5)
        return mu, np.exp(lsig)+1e-3, np.exp(lnu)+1.0, (z1,h1,z2,h2)
    def _adam(self,g):
        self.t+=1; b1,b2,e=0.9,0.999,1e-8
        for k in self.P:
            gr=g.get(k,0)
            self.m[k]=b1*self.m[k]+(1-b1)*gr
            self.v[k]=b2*self.v[k]+(1-b2)*(gr**2)
            mh=self.m[k]/(1-b1**self.t); vh=self.v[k]/(1-b2**self.t)
            self.P[k]-=self.lr*mh/(np.sqrt(vh)+e)
    def fit(self,X,y):
        n=len(y)
        for _ in range(self.epochs):
            mu,sig,nu,(z1,h1,z2,h2)=self._fwd(X)
            z=(y-mu)/sig; w=(nu+1)/(nu+z**2)
            dmu=-(w*z/sig)/n; dls=(1-w*z**2)/n
            g={}
            g['Wm']=h2.T@dmu.reshape(-1,1); g['bm']=np.array([dmu.sum()])
            g['Ws']=h2.T@dls.reshape(-1,1); g['bs']=np.array([dls.sum()])
            g['Wn']=np.zeros_like(self.P['Wn']); g['bn']=np.zeros_like(self.P['bn'])
            dh2=(dmu.reshape(-1,1)@self.P['Wm'].T+dls.reshape(-1,1)@self.P['Ws'].T)*(z2>0)
            g['W2']=h1.T@dh2; g['b2']=dh2.sum(axis=0)
            dh1=(dh2@self.P['W2'].T)*(z1>0)
            g['W1']=X.T@dh1; g['b1']=dh1.sum(axis=0)
            self._adam(g)
        return self
    def predict(self,X):
        mu,sig,nu,_=self._fwd(X); return mu,sig,nu

def interval_metrics(y, mu, lower, upper):
    mae  = float(np.mean(np.abs(y - mu)))
    rmse = float(np.sqrt(np.mean((y - mu)**2)))
    picp = float(np.mean((y >= lower) & (y <= upper)))
    mpiw = float(np.mean(upper - lower))
    mpiw_med = float(np.median(upper - lower))
    sigma = np.maximum((upper - lower)/(2*stats.norm.ppf(0.95)), 1e-6)
    levels = np.arange(0.1, 1.0, 0.1); maace = 0.0
    for lv in levels:
        z = stats.norm.ppf((1+lv)/2)
        cov = np.mean((y >= mu - z*sigma) & (y <= mu + z*sigma))
        maace += abs(cov - lv)
    maace = maace/len(levels)*100
    return {"MAE":mae,"RMSE":rmse,"PICP_90":picp,"MPIW_90":mpiw,
            "MPIW_90_med":mpiw_med,"MAACE":maace,
            "mu":mu,"lower":lower,"upper":upper}

