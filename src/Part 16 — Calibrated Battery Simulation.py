import os
import json
import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
from setup_imports import DATA_DIR
from german_epf_research import signed_log, inv_signed_log
import plotly.graph_objects as go
from plotly.subplots import make_subplots

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
            g['W1']=X.T@dh1; g['b1']=dh1.sum(0)
            self._adam(g)
        return self
    def predict(self,X):
        mu,sig,nu,_=self._fwd(X); return mu,sig,nu


def make_calibrated_predictions(df, features):
    train = df[df['datetime'] <= '2024-04-30 23:00'].copy()
    val   = df[(df['datetime'] >= '2024-05-01') & (df['datetime'] <= '2025-04-30 23:00')].copy()

    sX = StandardScaler(); X_tr = sX.fit_transform(train[features].values)
    z_tr = signed_log(train['price'].values)
    sz = StandardScaler(); z_tr_s = sz.fit_transform(z_tr.reshape(-1,1)).flatten()

    print("  Training Student-t (Adam) for the battery...")
    st = StudentTDNN_Adam(len(features), hidden=[64,32], lr=0.01, epochs=250)
    st.fit(X_tr, z_tr_s)

    def predict_z_sigma(frame):
        X = sX.transform(frame[features].values)
        mu_s, sig_s, nu = st.predict(X)
        mu_z = sz.inverse_transform(mu_s.reshape(-1,1)).flatten()
        sig_z = sig_s * sz.scale_[0]
        return mu_z, sig_z

    zt_val, sig_val = predict_z_sigma(val)
    z_val = signed_log(val['price'].values)
    norm_res = np.abs(z_val - zt_val) / np.maximum(sig_val, 1e-6)
    q90 = np.quantile(norm_res, 0.90)

    zt_all, sig_all = predict_z_sigma(df)
    mu_eur = inv_signed_log(zt_all)
    
    half = q90 * np.maximum(sig_all, 1e-6)
    hi = inv_signed_log(zt_all + half); lo = inv_signed_log(zt_all - half)
    sigma_eur = (hi - lo) / (2 * 1.28)
    sigma_eur = np.clip(sigma_eur, 1.0, 400.0)
    return mu_eur, sigma_eur


def run_battery_per_regime(mu, sigma, price, regime, xi=0.90, k=0.1):
    n_days = len(mu)//24
    pn={0:0.,1:0.,2:0.}; ps={0:0.,1:0.,2:0.}; days={0:0,1:0,2:0}
    skipped={0:0,1:0,2:0}
    for d in range(n_days):
        sl = slice(d*24,(d+1)*24)
        mu_d, sig_d, true_d = mu[sl], sigma[sl], price[sl]
        reg_d = regime[sl]
        if len(mu_d) < 24: continue
        rg = int(np.bincount(reg_d).argmax()); days[rg]+=1
        b = int(np.argmin(mu_d)); s = int(np.argmax(mu_d))
        if s <= b: b,s = min(b,s),max(b,s)
        realised = true_d[s]*xi - true_d[b]
        pn[rg] += realised
        gap = mu_d[s] - mu_d[b]
        doubt = k * np.sqrt(sig_d[b]**2 + sig_d[s]**2)
        if gap > doubt:
            ps[rg] += realised
        else:
            skipped[rg] += 1
    names={0:'Normal',1:'Elevated',2:'Crisis'}; rows=[]
    for r in [0,1,2]:
        dd=max(days[r],1)
        rows.append((names[r], pn[r]/dd, ps[r]/dd, days[r], skipped[r]))
    return rows


def plot_battery_plotly(rows, save_path=os.path.join(DATA_DIR, "part16_battery_calibrated.html")):
    print("\nGenerating interactive Part 16 battery Plotly dashboard...")
    regs = [r[0] for r in rows]
    A = [r[1] for r in rows]
    B = [r[2] for r in rows]
    rcol = {'Normal': '#1D9E75', 'Elevated': '#BA7517', 'Crisis': '#D85A30'}

    fig = make_subplots(
        rows=1, cols=2,
        subplot_titles=(
            "<b>Per-Regime Battery Profit (€ per MWh per day)</b>",
            "<b>Annual Extra Value Across Germany's 10 GWh Fleet</b>"
        ),
        horizontal_spacing=0.12
    )

    fig.add_trace(go.Bar(name='Naive (always trade)', x=regs, y=A, marker_color='#888780', text=[f"€{v:.0f}" for v in A], textposition='outside'), row=1, col=1)
    fig.add_trace(go.Bar(name='Smart (trade when sure)', x=regs, y=B, marker_color=[rcol[r] for r in regs], text=[f"€{v:.0f}" for v in B], textposition='outside'), row=1, col=1)

    fleet = 10000
    fv = [(b - a) * 365 * fleet / 1e6 for _, a, b, _, _ in rows]
    max_fv = max(fv) if fv else 10
    
    fig.add_trace(
        go.Bar(
            name='Extra Fleet Value', x=regs, y=fv,
            marker_color=[rcol[r] for r in regs],
            text=[f"€{v:.0f}M" for v in fv],
            textposition='outside', showlegend=False,
            hovertemplate="<b>%{x}</b><br>Extra Value: €%{y:.1f}M/year<extra></extra>"
        ),
        row=1, col=2
    )

    fig.update_layout(
        title=dict(text="<b>Part 16 — Money Story on the Best Model: Calibrated Student-t + Per-Regime Battery</b>", font=dict(size=16), x=0.5, xanchor="center"),
        paper_bgcolor="#f7f6f3", plot_bgcolor="#ffffff",
        barmode='group', height=650, width=1400,
        legend=dict(orientation="h", yanchor="bottom", y=1.12, xanchor="center", x=0.5)
    )

    fig.update_yaxes(title_text="Battery Profit (€ / MWh / day)", row=1, col=1)
    fig.update_yaxes(title_text="Extra Value per Year (€ Million)", row=1, col=2, range=[0, max_fv * 1.25])

    fig.write_html(save_path)
    print(f"  ✓ Saved interactive Part 16 dashboard to {save_path}")
    fig.show()


if __name__ == "__main__":
    df = pd.read_pickle(os.path.join(DATA_DIR, "data_part2.pkl"))
    with open(os.path.join(DATA_DIR, "selected_part4.json")) as f:
        features = json.load(f)

    dfr = pd.read_pickle(os.path.join(DATA_DIR, "data_part7_regimes.pkl"))[['datetime','regime']].drop_duplicates('datetime')
    df = df.merge(dfr, on='datetime', how='left')
    df['regime'] = df['regime'].fillna(0).astype(int)
    df = df.sort_values('datetime').reset_index(drop=True)

    print("\nGenerating calibrated predictions over full history...")
    mu, sigma = make_calibrated_predictions(df, features)
    price = df['price'].values
    regime = df['regime'].values

    print("\nRunning per-regime battery (naive vs smart)...")
    rows = run_battery_per_regime(mu, sigma, price, regime)

    print("\n" + "="*74)
    print("  PART 16 — PER-REGIME BATTERY ON THE CALIBRATED STUDENT-T")
    print("="*74)
    print(f"  {'Regime':<10}{'Naive €/day':>13}{'Smart €/day':>13}{'Extra':>9}{'Days':>7}{'Skipped':>9}")
    print("  " + "-"*68)
    for nm, a, b, dd, sk in rows:
        print(f"  {nm:<10}{a:>13.2f}{b:>13.2f}{b-a:>9.2f}{dd:>7}{sk:>9}")
    print("="*74)
    
    fleet = 10000
    print("\n  ANNUAL FLEET VALUE (Germany 10 GWh by 2030):")
    for nm, a, b, dd, sk in rows:
        print(f"    {nm:<10}: +€{b-a:5.2f}/MWh/day → €{(b-a)*365*fleet/1e6:6.1f}M/yr extra")

    np.save(os.path.join(DATA_DIR, "part16_results.npy"), {'rows': rows}, allow_pickle=True)
    plot_battery_plotly(rows, os.path.join(DATA_DIR, "part16_battery_calibrated.html"))
    print("\n✓ Part 16 complete!")