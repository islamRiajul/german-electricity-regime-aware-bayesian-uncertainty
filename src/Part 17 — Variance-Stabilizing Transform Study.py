import os
import json
import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
from setup_imports import DATA_DIR
from german_epf_research import DDNN, EvDNN, VIDDNN
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


class TransformSignedLogThenStd:
    name = "B. signed-log then std"
    def _slog(self, p): return np.sign(p) * np.log1p(np.abs(p))
    def _islog(self, z): return np.sign(z) * np.expm1(np.abs(z))
    def fit(self, price):
        a = self._slog(price); self.mu = a.mean(); self.sd = a.std()
    def forward(self, price): return (self._slog(price) - self.mu) / self.sd
    def inverse(self, z): return self._islog(z * self.sd + self.mu)

class TransformStdThenAsinh:
    name = "C. std then asinh"
    def fit(self, price): self.a = price.mean(); self.b = price.std()
    def forward(self, price): return np.arcsinh((price - self.a) / self.b)
    def inverse(self, z): return np.sinh(z) * self.b + self.a

class TransformStdThenAsinhRobust:
    name = "D. std(robust) then asinh"
    def fit(self, price):
        self.a = np.median(price)
        mad = np.median(np.abs(price - self.a))
        self.b = mad * 1.4826 + 1e-6
    def forward(self, price): return np.arcsinh((price - self.a) / self.b)
    def inverse(self, z): return np.sinh(z) * self.b + self.a


def evaluate_transform_model(T, model_name, model_fn, train, val, test, features):
    T.fit(train['price'].values)
    z_tr  = T.forward(train['price'].values)
    z_val = T.forward(val['price'].values)
    y_te  = test['price'].values

    sX = StandardScaler()
    X_tr = sX.fit_transform(train[features].values)
    X_val= sX.transform(val[features].values)
    X_te = sX.transform(test[features].values)

    sz = StandardScaler()
    z_tr_s  = sz.fit_transform(z_tr.reshape(-1,1)).flatten()
    z_val_s = sz.transform(z_val.reshape(-1,1)).flatten()

    model = model_fn(X_tr.shape[1])
    if model_name == "StudentT":
        model.fit(X_tr, z_tr_s)
    else:
        model.fit(X_tr, z_tr_s, X_val, z_val_s, verbose=False)

    def to_euros(z_scaled):
        z = sz.inverse_transform(z_scaled.reshape(-1,1)).flatten()
        return T.inverse(z), z

    if model_name == "StudentT":
        z_pred_s = model.predict(X_te)[0]
    else:
        z_pred_s, _ = model.predict(X_te)
    mu_eur, z_pred = to_euros(z_pred_s)
    mu_eur = np.clip(mu_eur, -600.0, 1000.0)

    if model_name == "StudentT":
        z_val_pred_s = model.predict(X_val)[0]
    else:
        z_val_pred_s, _ = model.predict(X_val)
    _, z_val_pred = to_euros(z_val_pred_s)
    q = np.quantile(np.abs(z_val - z_val_pred), 0.90)
    lo_eur = T.inverse(z_pred - q); hi_eur = T.inverse(z_pred + q)
    lo2 = np.clip(np.minimum(lo_eur, hi_eur), -600.0, 1000.0)
    hi2 = np.clip(np.maximum(lo_eur, hi_eur), -600.0, 1000.0)

    return {"transform": T.name, "model": model_name,
            "MAE": float(np.mean(np.abs(y_te - mu_eur))),
            "RMSE": float(np.sqrt(np.mean((y_te - mu_eur)**2))),
            "PICP_90": float(np.mean((y_te >= lo2) & (y_te <= hi2))),
            "MPIW_med": float(np.median(hi2 - lo2))}


if __name__ == "__main__":
    df = pd.read_pickle(os.path.join(DATA_DIR, 'data_part2.pkl'))
    with open(os.path.join(DATA_DIR, 'selected_part4.json')) as f: features = json.load(f)

    train = df[df['datetime'] <= '2024-04-30 23:00'].copy()
    val   = df[(df['datetime'] >= '2024-05-01') & (df['datetime'] <= '2025-04-30 23:00')].copy()
    test  = df[df['datetime'] >= '2025-05-01'].copy()

    transforms = [TransformSignedLogThenStd(),
                  TransformStdThenAsinh(), TransformStdThenAsinhRobust()]
    models_dict = {
        "DDNN":    lambda dim: DDNN(dim, hidden=[128,64,32], lr=0.002, epochs=120),
        "EvDNN":   lambda dim: EvDNN(dim, hidden=[128,64,32], lr=0.001, epochs=120, lam=0.02),
        "VI-DDNN": lambda dim: VIDDNN(dim, hidden=[128,64], lr=0.002, epochs=120, n_samples=15),
        "StudentT":lambda dim: StudentTDNN_Adam(dim, hidden=[64,32], lr=0.01, epochs=150),
    }

    results = []
    for T in transforms:
        for model_name, model_fn in models_dict.items():
            print(f"Evaluating Transform: {T.name} | Model: {model_name} ...")
            results.append(evaluate_transform_model(T, model_name, model_fn, train, val, test, features))

    print("\n" + "="*74)
    print("  PART 17 — TRANSFORM × MODEL COMPARISON (test May'25–Apr'26)")
    print("="*74)
    print(f"  {'Transform':<26}{'Model':<10}{'MAE':>9}{'RMSE':>9}{'PICP':>8}")
    print("  " + "-"*68)
    for r in sorted(results, key=lambda r:(r['transform'], r['MAE'])):
        print(f"  {r['transform']:<26}{r['model']:<10}{r['MAE']:>9.2f}{r['RMSE']:>9.2f}{r['PICP_90']:>8.1%}")
    print("="*74)
    best = min(results, key=lambda r:r['MAE'])
    print(f"\n  Best overall: {best['model']} with {best['transform']} (MAE €{best['MAE']:.2f})")

    np.save(os.path.join(DATA_DIR, 'part17_all_models_results.npy'), {'results':results}, allow_pickle=True)
    print("\n✓ Part 17 complete")