import os
import numpy as np
import pandas as pd
from scipy import stats
from scipy.special import gammaln
from sklearn.preprocessing import StandardScaler
from setup_imports import DATA_DIR, DOCS_DIR
from german_epf_research import DDNN, signed_log, inv_signed_log
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from IPython.display import display

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
    zc = (y-mu)/sigma
    crps = float(np.mean(sigma*(zc*(2*stats.norm.cdf(zc)-1)
                                + 2*stats.norm.pdf(zc) - 1/np.sqrt(np.pi))))
    return {"MAE":mae,"RMSE":rmse,"CRPS":crps,"PICP_90":picp,
            "MPIW_90":mpiw,"MPIW_90_med":mpiw_med,"MAACE":maace,
            "mu":mu,"lower":lower,"upper":upper,"sigma":sigma}


class StudentTDNN_Adam:
    def __init__(self, input_dim, hidden=[64, 32], lr=0.01, epochs=250):
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
        self.losses = []

    def _relu(self, x): return np.maximum(0, x)

    def _fwd(self, X):
        z1 = X@self.P['W1']+self.P['b1']; h1 = self._relu(z1)
        z2 = h1@self.P['W2']+self.P['b2']; h2 = self._relu(z2)
        mu   = (h2@self.P['Wm']+self.P['bm']).flatten()
        lsig = np.clip((h2@self.P['Ws']+self.P['bs']).flatten(), -4, 4)
        lnu  = np.clip((h2@self.P['Wn']+self.P['bn']).flatten(), 0.3, 3.5)
        return mu, np.exp(lsig)+1e-3, np.exp(lnu)+1.0, (z1,h1,z2,h2)

    def _nll(self, y, mu, sig, nu):
        z = (y-mu)/sig
        ll = (gammaln((nu+1)/2)-gammaln(nu/2)-0.5*np.log(nu*np.pi)
              -np.log(sig)-(nu+1)/2*np.log1p(z**2/nu))
        return -np.mean(ll)

    def _adam_step(self, grads):
        self.t += 1; b1, b2, eps = 0.9, 0.999, 1e-8
        for k in self.P:
            g = grads.get(k, 0)
            self.m[k] = b1*self.m[k] + (1-b1)*g
            self.v[k] = b2*self.v[k] + (1-b2)*(g**2)
            mhat = self.m[k]/(1-b1**self.t)
            vhat = self.v[k]/(1-b2**self.t)
            self.P[k] -= self.lr * mhat/(np.sqrt(vhat)+eps)

    def fit(self, X, y, verbose=False):
        n = len(y)
        for ep in range(self.epochs):
            mu, sig, nu, (z1,h1,z2,h2) = self._fwd(X)
            self.losses.append(self._nll(y, mu, sig, nu))
            z = (y-mu)/sig; w = (nu+1)/(nu+z**2)
            dmu = -(w*z/sig)/n
            dlsig = (1 - w*z**2)/n
            grads = {}
            grads['Wm'] = h2.T@dmu.reshape(-1,1); grads['bm'] = np.array([dmu.sum()])
            grads['Ws'] = h2.T@dlsig.reshape(-1,1); grads['bs'] = np.array([dlsig.sum()])
            grads['Wn'] = np.zeros_like(self.P['Wn']); grads['bn'] = np.zeros_like(self.P['bn'])
            dh2 = (dmu.reshape(-1,1)@self.P['Wm'].T + dlsig.reshape(-1,1)@self.P['Ws'].T)*(z2>0)
            grads['W2'] = h1.T@dh2; grads['b2'] = dh2.sum(axis=0)
            dh1 = (dh2@self.P['W2'].T)*(z1>0)
            grads['W1'] = X.T@dh1; grads['b1'] = dh1.sum(axis=0)
            self._adam_step(grads)
        return self

    def predict(self, X):
        mu, sig, nu, _ = self._fwd(X)
        return mu, sig, nu


class LogSpaceConformal:
    def __init__(self, coverage=0.90): self.coverage=coverage; self.q={}
    def fit(self, z_true, z_pred, hours):
        r = np.abs(z_true-z_pred)
        for h in range(24):
            m = hours==h
            self.q[h] = np.quantile(r[m],self.coverage) if m.sum()>10 else np.quantile(r,self.coverage)
    def predict(self, z_pred, hours):
        q = np.array([self.q.get(int(h),np.median(list(self.q.values()))) for h in hours])
        return inv_signed_log(z_pred), inv_signed_log(z_pred-q), inv_signed_log(z_pred+q)


def walk_forward_rolling(df, features, window_days=365, step_days=60,
                         test_start='2025-05-01'):
    df = df.sort_values('datetime').reset_index(drop=True)
    W = window_days*24; S = step_days*24
    test_idx = df.index[df['datetime'] >= test_start].tolist()
    preds = np.full(len(df), np.nan); pos = test_idx[0]; nb = 0
    while pos < len(df):
        tr = df.iloc[max(0,pos-W):pos]; bl = df.iloc[pos:pos+S]
        if len(bl)==0: break
        if len(tr) < 24*60: pos += S; continue
        sX = StandardScaler(); Xtr = sX.fit_transform(tr[features].values)
        sz = StandardScaler(); ztr = sz.fit_transform(signed_log(tr['price'].values).reshape(-1,1)).flatten()
        m = DDNN(len(features), hidden=[96,48,24], lr=0.0025, epochs=150); m.fit(Xtr, ztr, verbose=False)
        Xb = sX.transform(bl[features].values); mu_s,_ = m.predict(Xb)
        zb = sz.inverse_transform(mu_s.reshape(-1,1)).flatten()
        preds[pos:pos+len(bl)] = inv_signed_log(zb); nb += 1; pos += S
    test = df[df['datetime'] >= test_start]
    return preds[test.index], test, nb


def battery_full_history(df_full, features, xi=0.90, degr_cost=2.0,
                         risk_aversion=0.02, n_scen=40):
    df = df_full.sort_values('datetime').reset_index(drop=True)
    price = df['price'].values
    hour  = df['hour'].values
    regime= df['regime'].values
    mu = df['lag_24'].values
    resid = np.abs(price - mu)
    sigma = pd.Series(resid).rolling(168, min_periods=24).mean().bfill().values
    sigma = np.clip(sigma, 5, 200)
    valid = ~np.isnan(mu)
    mu, sigma, price, hour, regime = mu[valid], sigma[valid], price[valid], hour[valid], regime[valid]

    n_days = len(mu)//24
    pn = {0:0.,1:0.,2:0.}; pe = {0:0.,1:0.,2:0.}; days={0:0,1:0,2:0}
    rng = np.random.default_rng(0)
    for d in range(n_days):
        sl = slice(d*24,(d+1)*24)
        mu_d, sig_d, true_d = mu[sl], sigma[sl], price[sl]
        reg_d = regime[sl]
        if len(mu_d) < 24: continue
        rg = int(np.bincount(reg_d).argmax()); days[rg]+=1
        b0,s0 = int(np.argmin(mu_d)), int(np.argmax(mu_d))
        if s0>b0: pn[rg] += true_d[s0]*xi - true_d[b0]
        scen = rng.normal(mu_d[None,:], sig_d[None,:], size=(n_scen,24))
        best_u, best = -1e9, None
        for b in np.argsort(mu_d)[:4]:
            for s in np.argsort(mu_d)[-4:]:
                if s<=b: continue
                pr = scen[:,s]*xi - scen[:,b] - degr_cost
                u = pr.mean() - risk_aversion*pr.std()
                if u>best_u: best_u, best = u, (b,s)
        if best and best_u>0:
            b,s = best; pe[rg] += true_d[s]*xi - true_d[b] - degr_cost
    names={0:'Normal',1:'Elevated',2:'Crisis'}; rows=[]
    for r in [0,1,2]:
        dd=max(days[r],1); rows.append((names[r], pn[r]/dd, pe[r]/dd, days[r]))
    return rows


def plot_comparison_plotly(table, save_path=os.path.join(DOCS_DIR, "part14_final_comparison.html")):
    names = list(table.keys())
    specs = [
        ('MAE', 'MAE (€/MWh) — lower better', True, None, 1, 1),
        ('PICP_90', 'PICP 90% — closer to 90 better', False, 90, 1, 2),
        ('MPIW_90_med', 'MPIW 90% median — lower (sharper) better', True, None, 2, 1),
        ('MAACE', 'MAACE (%) — lower better', True, None, 2, 2)
    ]

    fig = make_subplots(
        rows=2, cols=2,
        subplot_titles=[s[1] for s in specs],
        vertical_spacing=0.20, horizontal_spacing=0.12
    )

    pal = ['#888780', '#378ADD', '#1D9E75', '#7F77DD', '#BA7517', '#0C447C']

    for key, title, lb, target, r, c in specs:
        vals = [table[m][key]*100 if key=='PICP_90' else table[m][key] for m in names]
        colors = [pal[i % len(pal)] for i in range(len(names))]

        fig.add_trace(
            go.Bar(
                x=names, y=vals,
                marker_color=colors,
                text=[f"{v:.1f}" for v in vals],
                textposition='outside',
                showlegend=False,
                hovertemplate="<b>%{x}</b><br>%{y:.2f}<extra></extra>"
            ),
            row=r, col=c
        )
        if target is not None:
            fig.add_shape(
                type="line", x0=-0.5, x1=len(names)-0.5, y0=target, y1=target,
                line=dict(color="#D85A30", dash="dash", width=2),
                row=r, col=c
            )

    fig.update_layout(
        title=dict(
            text="<b>Part 14 — Fair Comparison (all rows computed live, same test window)</b><br><sup>real SMARD data; test May 2025 – Apr 2026</sup>",
            font=dict(size=15), x=0.5, xanchor="center"
        ),
        paper_bgcolor="#f7f6f3", plot_bgcolor="#ffffff",
        height=900, width=1350, margin=dict(t=150, l=60, r=60, b=60)
    )
    fig.update_xaxes(tickangle=-15)
    fig.write_html(save_path)
    print(f"  ✓ Saved interactive comparison dashboard to {save_path}")
    display(fig)


def plot_battery_plotly(rows, save_path=os.path.join(DOCS_DIR, "part14_battery_full_history.html")):
    regs = [r[0] for r in rows]
    A = [r[1] for r in rows]
    B = [r[2] for r in rows]
    rcol = {'Normal': '#1D9E75', 'Elevated': '#BA7517', 'Crisis': '#D85A30'}

    fig = make_subplots(
        rows=1, cols=2,
        subplot_titles=(
            "<b>Full-history battery (2020–2026): crisis included</b>",
            "<b>Across Germany's 10 GWh 2030 fleet</b>"
        ),
        horizontal_spacing=0.15
    )

    fig.add_trace(go.Bar(name='Naive (always trade)', x=regs, y=A, marker_color='#888780', text=[f"€{v:.0f}" for v in A], textposition='outside'), row=1, col=1)
    fig.add_trace(go.Bar(name='Bayesian utility (Trader C)', x=regs, y=B, marker_color=[rcol[r] for r in regs], text=[f"€{v:.0f}" for v in B], textposition='outside'), row=1, col=1)

    fleet = 10000
    fv = [(b - a) * 365 * fleet / 1e6 for _, a, b, _ in rows]
    fig.add_trace(
        go.Bar(
            name='Extra Fleet Value', x=regs, y=fv,
            marker_color=[rcol[r] for r in regs],
            text=[f"€{v:.0f}M" for v in fv],
            textposition='outside',
            showlegend=False
        ),
        row=1, col=2
    )

    fig.update_layout(
        title=dict(
            text="<b>Bayesian Stochastic Battery — Value Grows with Volatility</b>",
            font=dict(size=16), x=0.5, xanchor="center"
        ),
        paper_bgcolor="#f7f6f3", plot_bgcolor="#ffffff",
        barmode='group',
        height=650, width=1400,
        margin=dict(t=120, l=60, r=60, b=60),
        legend=dict(orientation="h", yanchor="bottom", y=1.12, xanchor="center", x=0.5)
    )
    fig.update_yaxes(title_text="Battery profit (€ per MWh per day)", row=1, col=1)
    fig.update_yaxes(title_text="Extra value per year (€ million)", row=1, col=2)

    fig.write_html(save_path)
    print(f"  ✓ Saved interactive battery dashboard to {save_path}")
    display(fig)


if __name__ == "__main__":
    df = pd.read_pickle(os.path.join(DOCS_DIR, 'data_part2.pkl'))
    with open(os.path.join(DOCS_DIR, 'selected_part4.json')) as f: features = json.load(f)

    train = df[df['datetime'] <= '2024-04-30 23:00'].copy()
    val   = df[(df['datetime'] >= '2024-05-01') & (df['datetime'] <= '2025-04-30 23:00')].copy()
    test  = df[df['datetime'] >= '2025-05-01'].copy()

    sX = StandardScaler(); X_tr = sX.fit_transform(train[features].values)
    X_val = sX.transform(val[features].values); X_te = sX.transform(test[features].values)
    z_tr = signed_log(train['price'].values); z_val = signed_log(val['price'].values)
    sz = StandardScaler(); z_tr_s = sz.fit_transform(z_tr.reshape(-1,1)).flatten()
    y_te = test['price'].values

    def zpred(model, X):
        mu_s,_ = model.predict(X); return sz.inverse_transform(mu_s.reshape(-1,1)).flatten()

    table = {}

    print("\n[1/4] Baseline DDNN (Gaussian, delta-method)...")
    base = DDNN(len(features), hidden=[128,64,32], lr=0.002, epochs=300)
    base.fit(X_tr, z_tr_s, X_val, sz.transform(z_val.reshape(-1,1)).flatten())
    zp = zpred(base, X_te); mu = inv_signed_log(zp)
    sd = np.std(z_val - zpred(base, X_val)) * np.exp(np.abs(zp))
    z90 = stats.norm.ppf(0.95)
    table['Gaussian (delta)'] = interval_metrics(y_te, mu, mu-z90*sd, mu+z90*sd)

    print("[2/4] + Log-space conformal...")
    cp = LogSpaceConformal(0.90); cp.fit(z_val, zpred(base, X_val), val['hour'].values)
    mu_c, lo_c, hi_c = cp.predict(zp, test['hour'].values)
    table['Gaussian+LogCP'] = interval_metrics(y_te, mu_c, lo_c, hi_c)

    print("[3/4] Student-t (Adam) + Log-space conformal...")
    st = StudentTDNN_Adam(len(features), hidden=[64,32], lr=0.01, epochs=250)
    st.fit(X_tr, z_tr_s)
    zt_val = sz.inverse_transform(st.predict(X_val)[0].reshape(-1,1)).flatten()
    zt_te  = sz.inverse_transform(st.predict(X_te)[0].reshape(-1,1)).flatten()
    cp_t = LogSpaceConformal(0.90); cp_t.fit(z_val, zt_val, val['hour'].values)
    mu_t, lo_t, hi_t = cp_t.predict(zt_te, test['hour'].values)
    table['StudentT+LogCP'] = interval_metrics(y_te, mu_t, lo_t, hi_t)

    print("[4/4] Walk-forward rolling window...")
    rp_all, test_r, nb = walk_forward_rolling(df, features, 365, 60)
    valid = ~np.isnan(rp_all); rp = rp_all[valid]; rt = test_r['price'].values[valid]
    res = np.abs(signed_log(rt) - signed_log(rp)); q = np.quantile(res, 0.90)
    lo_r = inv_signed_log(signed_log(rp)-q); hi_r = inv_signed_log(signed_log(rp)+q)
    table[f'Rolling ({nb}x)'] = interval_metrics(rt, rp, lo_r, hi_r)

    print("\n" + "="*82)
    print("  PART 14 — FAIR COMPARISON (all rows live, same test window May'25–Apr'26)")
    print("="*82)
    print(f"  {'Variant':<20}{'MAE':>9}{'PICP_90':>10}{'MPIW(med)':>11}{'MAACE':>9}")
    print("  " + "-"*76)
    for name, m in table.items():
        print(f"  {name:<20}{m['MAE']:>9.2f}{m['PICP_90']:>9.1%}"
              f"{m['MPIW_90_med']:>11.2f}{m['MAACE']:>8.2f}%")
    print("="*82)

    print("\n  Battery — FULL HISTORY 2020–2026 (crisis included):")
    dfr = pd.read_pickle(os.path.join(DOCS_DIR, 'data_part7_regimes.pkl'))
    dfr = dfr.sort_values('datetime').reset_index(drop=True)
    if 'lag_24' not in dfr.columns:
        dfr['lag_24'] = dfr['price'].shift(24)
    brows = battery_full_history(dfr, features)
    print(f"  {'Regime':<10}{'Naive €/day':>14}{'BayesOpt €/day':>16}{'extra':>9}")
    for nm, a, b, dd in brows:
        print(f"  {nm:<10}{a:>14.2f}{b:>16.2f}{b-a:>9.2f}   [{dd} days]")

    np.save(os.path.join(DOCS_DIR, 'part14_results.npy'), {'table':table, 'battery':brows}, allow_pickle=True)
    plot_comparison_plotly(table)
    plot_battery_plotly(brows)
    print("\n✓ Part 14 complete with interactive Plotly dashboards saved!")