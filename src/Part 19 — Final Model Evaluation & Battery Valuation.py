import os
import json
import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
from setup_imports import DATA_DIR
from german_epf_research import DDNN, EvDNN, VIDDNN
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from IPython.display import display

class AsinhTransform:
    def fit(self, p): self.a=float(np.mean(p)); self.b=float(np.std(p)+1e-8)
    def forward(self, p): return np.arcsinh((p-self.a)/self.b)
    def inverse(self, z): return np.sinh(z)*self.b+self.a


class StudentTAdam:
    def __init__(self, d, hidden=(64,32), lr=0.01, epochs=250):
        h1,h2=hidden; self.lr,self.epochs=lr,epochs
        self.P={'W1':np.random.randn(d,h1)*np.sqrt(2/d),'b1':np.zeros(h1),
                'W2':np.random.randn(h1,h2)*np.sqrt(2/h1),'b2':np.zeros(h2),
                'Wm':np.random.randn(h2,1)*0.01,'bm':np.zeros(1),
                'Ws':np.random.randn(h2,1)*0.01,'bs':np.zeros(1),
                'Wn':np.random.randn(h2,1)*0.01,'bn':np.zeros(1)}
        self.m={k:np.zeros_like(v) for k,v in self.P.items()}
        self.v={k:np.zeros_like(v) for k,v in self.P.items()}; self.t=0
    def _relu(self,x): return np.maximum(0,x)
    def _fwd(self,X):
        z1=X@self.P['W1']+self.P['b1']; h1=self._relu(z1)
        z2=h1@self.P['W2']+self.P['b2']; h2=self._relu(z2)
        mu=(h2@self.P['Wm']+self.P['bm']).flatten()
        lsig=np.clip((h2@self.P['Ws']+self.P['bs']).flatten(),-4,4)
        lnu=np.clip((h2@self.P['Wn']+self.P['bn']).flatten(),0.3,3.5)
        return mu,np.exp(lsig)+1e-3,np.exp(lnu)+1.0,(z1,h1,z2,h2)
    def _adam(self,g):
        self.t+=1; b1,b2,e=0.9,0.999,1e-8
        for k in self.P:
            gr=g.get(k,0)
            self.m[k]=b1*self.m[k]+(1-b1)*gr; self.v[k]=b2*self.v[k]+(1-b2)*gr**2
            mh=self.m[k]/(1-b1**self.t); vh=self.v[k]/(1-b2**self.t)
            self.P[k]-=self.lr*mh/(np.sqrt(vh)+e)
    def fit(self,X,y):
        n=len(y)
        for _ in range(self.epochs):
            mu,sig,nu,(z1,h1,z2,h2)=self._fwd(X)
            z=(y-mu)/sig; w=(nu+1)/(nu+z**2)
            dmu=-(w*z/sig)/n; dls=(1-w*z**2)/n
            g={'Wm':h2.T@dmu.reshape(-1,1),'bm':np.array([dmu.sum()]),
               'Ws':h2.T@dls.reshape(-1,1),'bs':np.array([dls.sum()]),
               'Wn':np.zeros_like(self.P['Wn']),'bn':np.zeros_like(self.P['bn'])}
            dh2=(dmu.reshape(-1,1)@self.P['Wm'].T+dls.reshape(-1,1)@self.P['Ws'].T)*(z2>0)
            g['W2']=h1.T@dh2; g['b2']=dh2.sum(0)
            dh1=(dh2@self.P['W2'].T)*(z1>0)
            g['W1']=X.T@dh1; g['b1']=dh1.sum(0)
            self._adam(g)
        return self
    def predict(self,X):
        mu,sig,nu,_=self._fwd(X); return mu,sig,nu


class MultiLevelConformal:
    def __init__(self): self.q={}
    def fit(self,z_true,z_pred,sigma):
        r=np.abs(z_true-z_pred)/np.maximum(sigma,1e-6)
        for lv in np.arange(0.1,1.0,0.1): self.q[round(lv,1)]=np.quantile(r,lv)
    def band(self,z_pred,sigma,level,T):
        half=self.q[round(level,1)]*np.maximum(sigma,1e-6)
        return T.inverse(z_pred-half), T.inverse(z_pred+half)


if __name__ == "__main__":
    df = pd.read_pickle(os.path.join(DATA_DIR, "data_part2.pkl"))
    with open(os.path.join(DATA_DIR, "selected_part4.json")) as f: selected_features = json.load(f)
    with open(os.path.join(DATA_DIR, "features_part2.json"))  as f: all_features = json.load(f)

    DEFAULTS = {"DDNN":{"lr":0.005,"epochs":250,"batch_size":128},
                "EvDNN":{"lr":0.005,"epochs":250,"batch_size":128,"lam":0.02},
                "VI-DDNN":{"lr":0.002,"epochs":250,"batch_size":128,"prior_sigma":1.0}}
    try:
        with open(os.path.join(DATA_DIR, "results_part8.json")) as f:
            loaded = json.load(f)
        BP = {m: {**DEFAULTS[m], **loaded.get(m, {})} for m in DEFAULTS}
    except FileNotFoundError:
        BP = DEFAULTS
    STUDENTT_EPOCHS = 250

    train = df[df["datetime"] <= "2024-04-30 23:00"].copy()
    val   = df[(df["datetime"] >= "2024-05-01") & (df["datetime"] <= "2025-04-30 23:00")].copy()
    test  = df[df["datetime"] >= "2025-05-01"].copy()
    test_dates = test["datetime"].values

    def run_best_model(features, tag):
        T = AsinhTransform(); T.fit(train["price"].values)
        sX = StandardScaler()
        X_tr = sX.fit_transform(train[features].values)
        X_val = sX.transform(val[features].values)
        X_te = sX.transform(test[features].values)
        z_tr = T.forward(train["price"].values); z_val = T.forward(val["price"].values)
        sz = StandardScaler(); z_tr_s = sz.fit_transform(z_tr.reshape(-1,1)).flatten()
        y_te = test["price"].values
        model = StudentTAdam(len(features), hidden=(64,32), lr=0.01, epochs=STUDENTT_EPOCHS).fit(X_tr, z_tr_s)
        def pzs(X):
            mu_s, sig_s, nu = model.predict(X)
            return sz.inverse_transform(mu_s.reshape(-1,1)).flatten(), sig_s*sz.scale_[0]
        zt_val, sig_val = pzs(X_val); zt_te, sig_te = pzs(X_te)
        cal = MultiLevelConformal(); cal.fit(z_val, zt_val, sig_val)
        mu_eur = np.clip(T.inverse(zt_te), -600.0, 1000.0); lo90, hi90 = cal.band(zt_te, sig_te, 0.9, T)
        lo90 = np.clip(lo90, -600.0, 1000.0); hi90 = np.clip(hi90, -600.0, 1000.0)
        mae=float(np.mean(np.abs(y_te-mu_eur))); rmse=float(np.sqrt(np.mean((y_te-mu_eur)**2)))
        picp=float(np.mean((y_te>=np.minimum(lo90,hi90))&(y_te<=np.maximum(lo90,hi90))))
        mpiw=float(np.median(np.abs(hi90-lo90)))
        maace=0.0
        for lv in np.arange(0.1,1.0,0.1):
            lo,hi=cal.band(zt_te,sig_te,lv,T)
            maace+=abs(np.mean((y_te>=np.minimum(lo,hi))&(y_te<=np.maximum(lo,hi)))-lv)
        maace=maace/9*100
        return {"tag":tag,"MAE":mae,"RMSE":rmse,"PICP":picp,"MPIW":mpiw,"MAACE":maace,
                "mu":mu_eur,"lo":lo90,"hi":hi90,"y":y_te}

    res_sel = run_best_model(selected_features, f"Selected ({len(selected_features)})")
    res_all = run_best_model(all_features, f"All ({len(all_features)})")

    print("\n"+"="*72)
    print("  FINAL MODEL — BOTH FEATURE SETS (Student-t + asinh + multi-level CP)")
    print("="*72)
    print(f"  {'Feature set':<18}{'MAE':>9}{'RMSE':>9}{'PICP':>8}{'MPIW(med)':>11}{'MAACE':>9}")
    print("  "+"-"*66)
    for r in [res_sel, res_all]:
        print(f"  {r['tag']:<18}{r['MAE']:>9.2f}{r['RMSE']:>9.2f}{r['PICP']:>8.1%}{r['MPIW']:>11.2f}{r['MAACE']:>8.2f}%")
    print("="*72)
    winner = min([res_sel,res_all], key=lambda r:r['MAE'])
    print(f"  Winner: {winner['tag']} feature set (MAE €{winner['MAE']:.2f})")
    print("\n✓ Part 19 complete!")