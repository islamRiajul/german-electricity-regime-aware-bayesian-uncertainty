import os
import json
import numpy as np
import pandas as pd
from scipy import stats
from sklearn.preprocessing import StandardScaler
from setup_imports import DATA_DIR
from german_epf_research import signed_log, inv_signed_log
import plotly.graph_objects as go
from plotly.subplots import make_subplots

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


class NormalizedConformal:
    def __init__(self, coverage=0.90):
        self.coverage = coverage
        self.q = None
    def fit(self, z_true, z_pred, sigma):
        sigma = np.maximum(sigma, 1e-6)
        norm_res = np.abs(z_true - z_pred) / sigma
        self.q = np.quantile(norm_res, self.coverage)
    def predict(self, z_pred, sigma):
        sigma = np.maximum(sigma, 1e-6)
        half = self.q * sigma
        mu  = inv_signed_log(z_pred)
        lo  = inv_signed_log(z_pred - half)
        hi  = inv_signed_log(z_pred + half)
        return mu, lo, hi


class PlainConformal:
    def __init__(self, coverage=0.90): self.coverage=coverage; self.q=None
    def fit(self, z_true, z_pred, sigma=None):
        self.q = np.quantile(np.abs(z_true - z_pred), self.coverage)
    def predict(self, z_pred, sigma=None):
        return (inv_signed_log(z_pred),
                inv_signed_log(z_pred - self.q),
                inv_signed_log(z_pred + self.q))


def plot_calibration_fix_plotly(before, after, save_path=os.path.join(DATA_DIR, "part15_calibration_fix.html")):
    print("\nGenerating interactive Part 15 calibration fix Plotly dashboard...")
    fig = make_subplots(
        rows=1, cols=2,
        subplot_titles=(
            "<b>1. Reliability Calibration Curves</b>",
            "<b>2. Before vs. After 4-Metric Comparison</b>"
        ),
        horizontal_spacing=0.12
    )

    levels = np.arange(0.1, 1.0, 0.1) * 100
    for res, name, c in [(before, 'Before (Plain Conformal)', '#888780'),
                         (after, 'After (Scale-Aware Conformal)', '#1D9E75')]:
        mu, lo, hi = res['mu'], res['lower'], res['upper']
        sig = np.maximum((hi - lo) / (2 * stats.norm.ppf(0.95)), 1e-6)
        covs = []
        for lv in np.arange(0.1, 1.0, 0.1):
            z = stats.norm.ppf((1 + lv) / 2)
            covs.append(np.mean((res['_y'] >= mu - z * sig) & (res['_y'] <= mu + z * sig)) * 100)
        
        fig.add_trace(
            go.Scatter(x=levels, y=covs, mode='lines+markers', name=name, line=dict(color=c, width=2)),
            row=1, col=1
        )

    fig.add_trace(
        go.Scatter(x=[10, 90], y=[10, 90], mode='lines', line=dict(color='black', dash='dash', width=1.5), name='Ideal Calibration', showlegend=False),
        row=1, col=1
    )

    metrics = ['MAE', 'PICP_90', 'MPIW_90_med', 'MAACE']
    labels = ['MAE (€)', 'PICP (%)', 'MPIW med (€)', 'MAACE (%)']
    bvals = [before['MAE'], before['PICP_90']*100, before['MPIW_90_med'], before['MAACE']]
    avals = [after['MAE'], after['PICP_90']*100, after['MPIW_90_med'], after['MAACE']]

    fig.add_trace(go.Bar(x=labels, y=bvals, name='Before', marker_color='#888780', text=[f"{v:.1f}" for v in bvals], textposition='outside'), row=1, col=2)
    fig.add_trace(go.Bar(x=labels, y=avals, name='After', marker_color='#1D9E75', text=[f"{v:.1f}" for v in avals], textposition='outside'), row=1, col=2)

    fig.update_layout(
        title=dict(text="<b>Part 15 — Fixing Student-t Calibration with Scale-Aware Conformal</b>", font=dict(size=16), x=0.5, xanchor="center"),
        paper_bgcolor="#f7f6f3", plot_bgcolor="#ffffff",
        barmode='group', height=650, width=1400,
        legend=dict(orientation="h", yanchor="bottom", y=1.12, xanchor="center", x=0.5)
    )

    fig.update_xaxes(title_text="Claimed Confidence (%)", row=1, col=1)
    fig.update_yaxes(title_text="Actual Coverage (%)", row=1, col=1)
    fig.update_yaxes(title_text="Metric Score", row=1, col=2)

    fig.write_html(save_path)
    print(f"  ✓ Saved interactive calibration dashboard to {save_path}")
    fig.show()


if __name__ == "__main__":
    df = pd.read_pickle(os.path.join(DATA_DIR, "data_part2.pkl"))
    with open(os.path.join(DATA_DIR, "selected_part4.json")) as f: features = json.load(f)

    train = df[df['datetime'] <= '2024-04-30 23:00'].copy()
    val   = df[(df['datetime'] >= '2024-05-01') & (df['datetime'] <= '2025-04-30 23:00')].copy()
    test  = df[df['datetime'] >= '2025-05-01'].copy()

    sX = StandardScaler(); X_tr = sX.fit_transform(train[features].values)
    X_val = sX.transform(val[features].values); X_te = sX.transform(test[features].values)
    z_tr = signed_log(train['price'].values); z_val = signed_log(val['price'].values)
    sz = StandardScaler(); z_tr_s = sz.fit_transform(z_tr.reshape(-1,1)).flatten()
    y_te = test['price'].values

    print("\nTraining Student-t (Adam)...")
    st = StudentTDNN_Adam(len(features), hidden=[64,32], lr=0.01, epochs=250)
    st.fit(X_tr, z_tr_s)

    def predict_z_sigma(X):
        mu_s, sig_s, nu = st.predict(X)
        mu_z = sz.inverse_transform(mu_s.reshape(-1,1)).flatten()
        sig_z = sig_s * sz.scale_[0]
        return mu_z, sig_z
    
    zt_val, sig_val = predict_z_sigma(X_val)
    zt_te,  sig_te  = predict_z_sigma(X_te)

    pc = PlainConformal(0.90); pc.fit(z_val, zt_val)
    mu_b, lo_b, hi_b = pc.predict(zt_te)
    before = interval_metrics(y_te, mu_b, lo_b, hi_b); before['_y'] = y_te

    nc = NormalizedConformal(0.90); nc.fit(z_val, zt_val, sig_val)
    mu_a, lo_a, hi_a = nc.predict(zt_te, sig_te)
    after = interval_metrics(y_te, mu_a, lo_a, hi_a); after['_y'] = y_te

    print("\n" + "="*70)
    print("  CALIBRATION FIX — BEFORE vs AFTER (same Student-t model)")
    print("="*70)
    print(f"  {'Metric':<14}{'BEFORE (plain)':>16}{'AFTER (scale-aware)':>22}")
    print("  " + "-"*54)
    for k, lab in [('MAE','MAE (€)'),('PICP_90','PICP 90%'),
                   ('MPIW_90_med','MPIW median (€)'),('MAACE','MAACE (%)')]:
        fb = (lambda v: f"{v:.1%}") if k=='PICP_90' else (lambda v: f"{v:.2f}")
        print(f"  {lab:<14}{fb(before[k]):>16}{fb(after[k]):>22}")
    print("="*70)

    np.save(os.path.join(DATA_DIR, "part15_results.npy"), {'before': before, 'after': after}, allow_pickle=True)
    plot_calibration_fix_plotly(before, after, os.path.join(DATA_DIR, "part15_calibration_fix.html"))
    print("\n✓ Part 15 complete!")