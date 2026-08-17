import os
import json
import numpy as np
import pandas as pd
from scipy import stats
from sklearn.preprocessing import StandardScaler
from setup_imports import DATA_DIR, DOCS_DIR
from german_epf_research import DDNN, signed_log, inv_signed_log
import plotly.graph_objects as go
from plotly.subplots import make_subplots

class SignedLogConformal:
    def __init__(self, coverage=0.90):
        self.coverage = coverage
        self.q_by_hour = {}

    def fit(self, z_true_cal, z_pred_cal, hours_cal):
        resid = np.abs(z_true_cal - z_pred_cal)
        for h in range(24):
            mask = hours_cal == h
            if mask.sum() > 10:
                self.q_by_hour[h] = np.quantile(resid[mask], self.coverage)
            else:
                self.q_by_hour[h] = np.quantile(resid, self.coverage)

    def predict(self, z_pred_test, hours_test):
        q = np.array([self.q_by_hour.get(int(h), np.median(list(self.q_by_hour.values())))
                      for h in hours_test])
        z_lower = z_pred_test - q
        z_upper = z_pred_test + q
        mu_eur    = inv_signed_log(z_pred_test)
        lower_eur = inv_signed_log(z_lower)
        upper_eur = inv_signed_log(z_upper)
        return mu_eur, lower_eur, upper_eur


def metrics_from_interval(y_true, mu, lower, upper, coverage=0.90):
    mae  = float(np.mean(np.abs(y_true - mu)))
    rmse = float(np.sqrt(np.mean((y_true - mu)**2)))
    picp = float(np.mean((y_true >= lower) & (y_true <= upper)))
    mpiw = float(np.mean(upper - lower))
    levels = np.arange(0.1, 1.0, 0.1)
    maace = 0.0
    half = (upper - lower) / 2
    z90 = stats.norm.ppf(0.95)
    sigma_equiv = half / z90
    for lv in levels:
        z = stats.norm.ppf((1+lv)/2)
        lo = mu - z*sigma_equiv; hi = mu + z*sigma_equiv
        cov = np.mean((y_true >= lo) & (y_true <= hi))
        maace += abs(cov - lv)
    maace = maace/len(levels)*100
    return {"MAE":mae, "RMSE":rmse, "PICP_90":picp, "MPIW_90":mpiw, "MAACE":maace}


def run_signed_log_conformal(train, val, test, features):
    print("\n" + "="*70)
    print("  IMPROVEMENT 1: SIGNED-LOG CONFORMAL FIX")
    print("="*70)

    z_tr  = signed_log(train['price'].values)
    z_val = signed_log(val['price'].values)
    y_te  = test['price'].values

    scaler_X = StandardScaler()
    X_tr = scaler_X.fit_transform(train[features].values)
    X_val= scaler_X.transform(val[features].values)
    X_te = scaler_X.transform(test[features].values)

    scaler_z = StandardScaler()
    z_tr_s  = scaler_z.fit_transform(z_tr.reshape(-1,1)).flatten()
    z_val_s = scaler_z.transform(z_val.reshape(-1,1)).flatten()

    print("  Training DDNN in signed-log space...")
    ddnn = DDNN(len(features), hidden=[128,64,32], lr=0.002, epochs=300)
    ddnn.fit(X_tr, z_tr_s, X_val, z_val_s)

    def predict_z(X):
        mu_s, _ = ddnn.predict(X)
        return scaler_z.inverse_transform(mu_s.reshape(-1,1)).flatten()
    z_pred_val  = predict_z(X_val)
    z_pred_test = predict_z(X_te)

    mu_eur = inv_signed_log(z_pred_test)
    sig_z  = np.std(z_val - z_pred_val)
    sig_eur_delta = sig_z * np.exp(np.abs(z_pred_test))
    z90 = stats.norm.ppf(0.95)
    old = metrics_from_interval(y_te, mu_eur,
                                mu_eur - z90*sig_eur_delta,
                                mu_eur + z90*sig_eur_delta)

    cp = SignedLogConformal(coverage=0.90)
    cp.fit(z_val, z_pred_val, val['hour'].values)
    mu_new, lo_new, hi_new = cp.predict(z_pred_test, test['hour'].values)
    new = metrics_from_interval(y_te, mu_new, lo_new, hi_new)

    print(f"\n  {'Metric':<10}{'OLD (delta)':>14}{'NEW (log-conformal)':>22}")
    print("  " + "-"*46)
    for k in ['MAE','PICP_90','MPIW_90','MAACE']:
        fmt = (lambda v: f"{v:.1%}") if k=='PICP_90' else (lambda v: f"{v:.2f}")
        print(f"  {k:<10}{fmt(old[k]):>14}{fmt(new[k]):>22}")

    return {'old':old, 'new':new, 'mu':mu_new, 'lower':lo_new, 'upper':hi_new, 'y_true':y_te}


def run_rolling_window(df, features, window_days=365, step_days=30):
    print("\n" + "="*70)
    print("  IMPROVEMENT 2: ROLLING-WINDOW RETRAINING")
    print("="*70)
    print(f"  Window: {window_days} days, retrain every {step_days} days")

    df = df.sort_values('datetime').reset_index(drop=True)
    test_start = pd.Timestamp('2025-05-01')
    test_idx = df.index[df['datetime'] >= test_start].tolist()

    hours_per_day = 24
    window = window_days * hours_per_day
    step   = step_days * hours_per_day

    preds_rolling = np.full(len(df), np.nan)
    start = test_idx[0]
    print("  Walking forward (retraining each block)...")
    n_blocks = 0
    pos = start
    while pos < len(df):
        tr_lo = max(0, pos - window)
        train_block = df.iloc[tr_lo:pos]
        pred_block  = df.iloc[pos:pos+step]
        if len(pred_block) == 0: break
        if len(train_block) < 24*30:
            pos += step; continue

        sX = StandardScaler()
        Xtr = sX.fit_transform(train_block[features].values)
        ztr = signed_log(train_block['price'].values)
        sz  = StandardScaler(); ztr_s = sz.fit_transform(ztr.reshape(-1,1)).flatten()

        model = DDNN(len(features), hidden=[128,64,32], lr=0.0025, epochs=200)
        model.fit(Xtr, ztr_s, verbose=False)

        Xpr = sX.transform(pred_block[features].values)
        mu_s, _ = model.predict(Xpr)
        z_pred = sz.inverse_transform(mu_s.reshape(-1,1)).flatten()
        preds_rolling[pos:pos+len(pred_block)] = inv_signed_log(z_pred)

        n_blocks += 1
        pos += step
    print(f"  Retrained {n_blocks} times across the test period")

    print("  Training the frozen (train-once) model for comparison...")
    pre = df[df['datetime'] < test_start]
    sX = StandardScaler(); Xtr = sX.fit_transform(pre[features].values)
    ztr = signed_log(pre['price'].values)
    sz = StandardScaler(); ztr_s = sz.fit_transform(ztr.reshape(-1,1)).flatten()
    frozen = DDNN(len(features), hidden=[128,64,32], lr=0.002, epochs=300)
    frozen.fit(Xtr, ztr_s, verbose=False)
    test_df = df[df['datetime'] >= test_start]
    Xte = sX.transform(test_df[features].values)
    mu_s,_ = frozen.predict(Xte)
    frozen_pred = inv_signed_log(sz.inverse_transform(mu_s.reshape(-1,1)).flatten())

    y_te = test_df['price'].values
    roll_pred = preds_rolling[test_df.index]
    valid = ~np.isnan(roll_pred)
    mae_frozen  = np.mean(np.abs(y_te[valid] - frozen_pred[valid]))
    mae_rolling = np.mean(np.abs(y_te[valid] - roll_pred[valid]))

    print(f"\n  {'Strategy':<22}{'MAE (€/MWh)':>14}")
    print("  " + "-"*36)
    print(f"  {'Frozen (train once)':<22}{mae_frozen:>14.2f}")
    print(f"  {'Rolling window':<22}{mae_rolling:>14.2f}")
    improvement = (mae_frozen - mae_rolling)/mae_frozen*100
    print(f"\n  Rolling window improves MAE by {improvement:.1f}%")

    return {'mae_frozen':mae_frozen, 'mae_rolling':mae_rolling,
            'improvement':improvement, 'y_true':y_te[valid],
            'frozen':frozen_pred[valid], 'rolling':roll_pred[valid]}


class SimpleBayesOpt:
    def __init__(self, bounds, n_init=5, n_iter=15):
        self.bounds = bounds
        self.names  = list(bounds)
        self.n_init = n_init
        self.n_iter = n_iter
        self.X = []
        self.y = []

    def _sample_random(self):
        return np.array([np.random.rand() for _ in self.names])

    def _to_real(self, x01):
        out = {}
        for i, n in enumerate(self.names):
            lo, hi = self.bounds[n]
            out[n] = lo + x01[i]*(hi-lo)
        return out

    def _surrogate(self, xq):
        if not self.X:
            return 0.0, 1.0
        X = np.array(self.X)
        d = np.linalg.norm(X - xq, axis=1)
        w = np.exp(-d**2 / 0.1)
        if w.sum() < 1e-9:
            return np.mean(self.y), 1.0
        mu = np.sum(w*np.array(self.y)) / w.sum()
        unc = 1.0 / (1.0 + w.sum())
        return mu, unc

    def _acquisition(self, xq):
        mu, unc = self._surrogate(xq)
        best = min(self.y) if self.y else 0
        return (best - mu) + 1.5*unc

    def optimize(self, objective):
        for _ in range(self.n_init):
            x = self._sample_random()
            score = objective(self._to_real(x))
            self.X.append(x); self.y.append(score)

        for it in range(self.n_iter):
            cands = [self._sample_random() for _ in range(200)]
            acq = [self._acquisition(c) for c in cands]
            x_next = cands[int(np.argmax(acq))]
            score = objective(self._to_real(x_next))
            self.X.append(x_next); self.y.append(score)

        best_i = int(np.argmin(self.y))
        return self._to_real(self.X[best_i]), self.y[best_i]


def run_bayesian_hpo(train, val, features):
    print("\n" + "="*70)
    print("  IMPROVEMENT 3: BAYESIAN HYPERPARAMETER OPTIMIZATION")
    print("="*70)

    z_tr  = signed_log(train['price'].values)
    z_val = signed_log(val['price'].values)
    sX = StandardScaler()
    tr_sub = train.iloc[::3]
    X_tr = sX.fit_transform(tr_sub[features].values)
    X_val= sX.transform(val[features].values)
    sz = StandardScaler()
    z_tr_s = sz.fit_transform(signed_log(tr_sub['price'].values).reshape(-1,1)).flatten()
    z_val_s= sz.transform(z_val.reshape(-1,1)).flatten()
    y_val = val['price'].values

    def objective(hp):
        h1 = int(hp['hidden1']); h2 = int(hp['hidden2']); h3 = max(8, h2//2)
        lr = hp['lr']
        model = DDNN(len(features), hidden=[h1, h2, h3], lr=lr, epochs=80)
        model.fit(X_tr, z_tr_s, verbose=False)
        mu_s, _ = model.predict(X_val)
        z_pred = sz.inverse_transform(mu_s.reshape(-1,1)).flatten()
        mu_eur = inv_signed_log(z_pred)
        return float(np.mean(np.abs(y_val - mu_eur)))

    bounds = {
        'hidden1': (32, 256),
        'hidden2': (16, 128),
        'lr':      (0.0005, 0.01),
    }
    print(f"  Searching: hidden1[32-256], hidden2[16-128], lr[0.0005-0.01]")
    print(f"  Running Bayesian optimization (4 random + 6 guided trials)...")

    opt = SimpleBayesOpt(bounds, n_init=4, n_iter=6)
    best_hp, best_score = opt.optimize(objective)

    baseline_mae = objective({'hidden1':128, 'hidden2':64, 'lr':0.002})
    print(f"\n  Hand-set baseline:  hidden=[128,64], lr=0.002  → val MAE €{baseline_mae:.2f}")
    print(f"  Bayesian-optimized: hidden=[{int(best_hp['hidden1'])},"
          f"{int(best_hp['hidden2'])}], lr={best_hp['lr']:.4f}  → val MAE €{best_score:.2f}")
    improvement = (baseline_mae - best_score)/baseline_mae*100
    print(f"  Improvement: {improvement:+.1f}%")

    return {'best_hp':best_hp, 'best_score':best_score,
            'baseline':baseline_mae, 'history':opt.y}


def plot_part13_dashboard(r1, r2, r3, save_path=os.path.join(DOCS_DIR, "part13_improvements_dashboard.html")):
    print("\nGenerating interactive Part 13 Plotly dashboard...")

    fig = make_subplots(
        rows=2, cols=2,
        subplot_titles=(
            "<b>1. Conformal Prediction: Old (Delta) vs. New (Log-Space)</b>",
            "<b>2. Retraining Strategy: Frozen vs. Rolling Window MAE</b>",
            "<b>3. Bayesian HPO: Validation MAE per Search Iteration</b>",
            "<b>4. Log-Conformal 90% Prediction Intervals (Test Slice)</b>"
        ),
        vertical_spacing=0.18,
        horizontal_spacing=0.10,
        specs=[[{"type": "bar"}, {"type": "bar"}],
               [{"type": "scatter"}, {"type": "scatter"}]]
    )

    metrics = ['MAE', 'PICP_90', 'MPIW_90', 'MAACE']
    old_vals = [r1['old'][k] * 100 if k == 'PICP_90' else r1['old'][k] for k in metrics]
    new_vals = [r1['new'][k] * 100 if k == 'PICP_90' else r1['new'][k] for k in metrics]

    fig.add_trace(go.Bar(x=metrics, y=old_vals, name='Old (Delta Method)', marker_color='#888780'), row=1, col=1)
    fig.add_trace(go.Bar(x=metrics, y=new_vals, name='New (Log-Conformal)', marker_color='#1D9E75'), row=1, col=1)

    strategies = ['Frozen (Train Once)', 'Rolling Window']
    maes = [r2['mae_frozen'], r2['mae_rolling']]
    fig.add_trace(go.Bar(
        x=strategies, y=maes,
        marker_color=['#888780', '#378ADD'],
        text=[f"€{v:.2f}" for v in maes],
        textposition='auto',
        showlegend=False
    ), row=1, col=2)

    history_scores = r3['history']
    iterations = list(range(1, len(history_scores) + 1))
    fig.add_trace(go.Scatter(
        x=iterations, y=history_scores,
        mode='lines+markers',
        name='Surrogate Validation MAE',
        line=dict(color='#D85A30', width=2),
        marker=dict(size=8),
        showlegend=False
    ), row=2, col=1)

    n_slice = min(168, len(r1['y_true']))
    x_idx = list(range(n_slice))
    y_true_slice = r1['y_true'][-n_slice:]
    mu_slice = r1['mu'][-n_slice:]
    lower_slice = r1['lower'][-n_slice:]
    upper_slice = r1['upper'][-n_slice:]

    fig.add_trace(go.Scatter(
        x=x_idx + x_idx[::-1],
        y=np.concatenate([upper_slice, lower_slice[::-1]]),
        fill='toself',
        fillcolor='rgba(29, 158, 117, 0.2)',
        line=dict(color='rgba(255,255,255,0)'),
        name='90% Conformal Interval',
        showlegend=True
    ), row=2, col=2)

    fig.add_trace(go.Scatter(
        x=x_idx, y=y_true_slice,
        mode='lines',
        name='Actual Price',
        line=dict(color='#333333', width=1)
    ), row=2, col=2)

    fig.add_trace(go.Scatter(
        x=x_idx, y=mu_slice,
        mode='lines',
        name='Predicted Mean',
        line=dict(color='#1D9E75', width=1.5)
    ), row=2, col=2)

    fig.update_layout(
        title=dict(
            text="<b>Part 13 — Three Model Improvements Interactive Dashboard</b>",
            font=dict(size=16), x=0.5, xanchor="center"
        ),
        paper_bgcolor="#f7f6f3",
        plot_bgcolor="#ffffff",
        height=850,
        width=1400,
        barmode='group',
        legend=dict(orientation="h", yanchor="bottom", y=1.04, xanchor="center", x=0.5)
    )

    fig.update_yaxes(title_text="Metric Value", row=1, col=1)
    fig.update_yaxes(title_text="MAE (€/MWh)", row=1, col=2)
    fig.update_xaxes(title_text="Optimization Trial", row=2, col=1)
    fig.update_yaxes(title_text="Validation MAE (€)", row=2, col=1)
    fig.update_xaxes(title_text="Test Hours (Slice)", row=2, col=2)
    fig.update_yaxes(title_text="Electricity Price (€/MWh)", row=2, col=2)

    fig.write_html(save_path)
    print(f"  ✓ Saved Part 13 interactive dashboard to {save_path}")
    fig.show()


if __name__ == "__main__":
    df = pd.read_pickle(os.path.join(DATA_DIR, "data_part2.pkl"))
    with open(os.path.join(DATA_DIR, "selected_part4.json")) as f:
        features = json.load(f)

    train = df[df['datetime'] <= '2024-04-30 23:00'].copy()
    val   = df[(df['datetime'] >= '2024-05-01') & (df['datetime'] <= '2025-04-30 23:00')].copy()
    test  = df[df['datetime'] >= '2025-05-01'].copy()

    r1 = run_signed_log_conformal(train, val, test, features)
    r3 = run_bayesian_hpo(train, val, features)
    r2 = run_rolling_window(df, features, window_days=365, step_days=30)

    np.save(os.path.join(DATA_DIR, "improvements.npy"),
            {'conformal':r1, 'rolling':r2, 'hpo':r3}, allow_pickle=True)
    
    plot_part13_dashboard(r1, r2, r3)

    print("\n" + "="*70)
    print("  ALL THREE IMPROVEMENTS COMPLETE (WITH PLOTLY DASHBOARD)")
    print("="*70)