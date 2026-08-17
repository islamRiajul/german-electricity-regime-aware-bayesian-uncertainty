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


class MultiLevelNormalizedConformal:
    def __init__(self):
        self.q = {}
    def fit(self, z_true, z_pred, sigma):
        sigma = np.maximum(sigma, 1e-6)
        nr = np.abs(z_true - z_pred) / sigma
        for lv in np.arange(0.1, 1.0, 0.1):
            self.q[round(lv, 1)] = np.quantile(nr, lv)
    def band(self, z_pred, sigma, level):
        half = self.q[round(level, 1)] * np.maximum(sigma, 1e-6)
        return inv_signed_log(z_pred - half), inv_signed_log(z_pred + half)


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
            "MPIW_90_med":mpiw_med,"MAACE":maace}


def plot_multilevel_calibration_plotly(mlc, y_te, zt_te, sig_te, m, maace, save_path=os.path.join(DATA_DIR, "part15b_multilevel_calibration.html")):
    print("\nGenerating interactive Multi-Level Conformal Plotly dashboard...")
    fig = make_subplots(
        rows=1, cols=2,
        subplot_titles=(
            "<b>1. Multi-Level Reliability Calibration Curve (Flat Calibration)</b>",
            "<b>2. Multi-Level Performance Metrics Summary</b>"
        ),
        horizontal_spacing=0.12
    )

    levels = np.arange(0.1, 1.0, 0.1) * 100
    empirical_covs = []
    for lv in np.arange(0.1, 1.0, 0.1):
        lo, hi = mlc.band(zt_te, sig_te, lv)
        cov = np.mean((y_te >= lo) & (y_te <= hi)) * 100
        empirical_covs.append(cov)

    fig.add_trace(
        go.Scatter(x=levels, y=empirical_covs, mode='lines+markers', name='Multi-Level Conformal', line=dict(color='#1D9E75', width=2.5), marker=dict(size=8)),
        row=1, col=1
    )
    fig.add_trace(
        go.Scatter(x=[10, 90], y=[10, 90], mode='lines', line=dict(color='black', dash='dash', width=1.5), name='Ideal Calibration', showlegend=False),
        row=1, col=1
    )

    metrics_labels = ['MAE (€)', 'PICP 90% (%)', 'MPIW Median (€)', 'MAACE (%)']
    metrics_values = [m['MAE'], m['PICP_90'] * 100, m['MPIW_90_med'], maace]
    fig.add_trace(
        go.Bar(x=metrics_labels, y=metrics_values, marker_color=['#378ADD', '#1D9E75', '#BA7517', '#D85A30'], text=[f"{v:.1f}" for v in metrics_values], textposition='outside', showlegend=False),
        row=1, col=2
    )

    fig.update_layout(
        title=dict(text="<b>Part 15b — Multi-Level Calibrated Student-t Dashboard</b>", font=dict(size=16), x=0.5, xanchor="center"),
        paper_bgcolor="#f7f6f3", plot_bgcolor="#ffffff",
        height=650, width=1400,
        legend=dict(orientation="h", yanchor="bottom", y=1.12, xanchor="center", x=0.5)
    )

    fig.update_xaxes(title_text="Claimed Confidence (%)", row=1, col=1)
    fig.update_yaxes(title_text="Actual Coverage (%)", row=1, col=1)
    fig.update_yaxes(title_text="Score / Value", row=1, col=2)

    fig.write_html(save_path)
    print(f"  ✓ Saved interactive multi-level dashboard to {save_path}")
    fig.show()


if __name__ == "__main__":
    df = pd.read_pickle(os.path.join(DATA_DIR, "data_part2.pkl"))
    with open(os.path.join(DATA_DIR, "selected_part4.json")) as f: features = json.load(f)

    train = df[df['datetime'] <= '2024-04-30 23:00'].copy()
    val   = df[(df['datetime'] >= '2024-05-01') & (df['datetime'] <= '2025-04-30 23:00')].copy()
    test  = df[df['datetime'] >= '2025-05-01'].copy()

    sX = StandardScaler()
    X_tr = sX.fit_transform(train[features].values)
    X_val = sX.transform(val[features].values)
    X_te = sX.transform(test[features].values)

    z_tr = signed_log(train['price'].values)
    z_val = signed_log(val['price'].values)
    sz = StandardScaler()
    z_tr_s = sz.fit_transform(z_tr.reshape(-1, 1)).flatten()
    y_te = test['price'].values

    print("Training Student-t...")
    st = StudentTDNN_Adam(len(features), hidden=[64, 32], lr=0.01, epochs=250)
    st.fit(X_tr, z_tr_s)

    def pzs(X):
        mu_s, sig_s, nu = st.predict(X)
        return sz.inverse_transform(mu_s.reshape(-1, 1)).flatten(), sig_s * sz.scale_[0]

    zt_val, sig_val = pzs(X_val)
    zt_te, sig_te = pzs(X_te)

    mlc = MultiLevelNormalizedConformal()
    mlc.fit(z_val, zt_val, sig_val)
    
    mu = inv_signed_log(zt_te)
    lo90, hi90 = mlc.band(zt_te, sig_te, 0.9)
    m = interval_metrics(y_te, mu, lo90, hi90)

    maace = 0.0
    for lv in np.arange(0.1, 1.0, 0.1):
        lo, hi = mlc.band(zt_te, sig_te, lv)
        cov = np.mean((y_te >= lo) & (y_te <= hi))
        maace += abs(cov - lv)
    maace = maace / 9 * 100

    print("\n" + "="*60)
    print("  MULTI-LEVEL CALIBRATED STUDENT-T (final)")
    print("="*60)
    print(f"  MAE           : €{m['MAE']:.2f}")
    print(f"  PICP 90%      : {m['PICP_90']:.1%}")
    print(f"  MPIW median   : €{m['MPIW_90_med']:.2f}")
    print(f"  MAACE (proper): {maace:.2f}%")
    print("="*60)

    np.save(os.path.join(DATA_DIR, "part15b_results.npy"), {'metrics': m, 'maace': maace}, allow_pickle=True)
    plot_multilevel_calibration_plotly(mlc, y_te, zt_te, sig_te, m, maace, os.path.join(DATA_DIR, "part15b_multilevel_calibration.html"))
    print("\n✓ Part 15b complete!")