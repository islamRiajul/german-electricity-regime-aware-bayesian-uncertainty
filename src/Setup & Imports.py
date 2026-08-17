import os
import json
import warnings
import numpy as np
import pandas as pd
from IPython.display import display
from scipy import stats
from scipy.special import gammaln
from scipy.stats import pearsonr, spearmanr
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import ElasticNet, Lasso, LinearRegression
from sklearn.feature_selection import mutual_info_regression
from sklearn.inspection import permutation_importance
import webbrowser
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from sklearn.feature_selection import (
    RFE,
    SequentialFeatureSelector,
    f_regression,
)
import plotly.io as pio
pio.renderers.default = "iframe"
# ── Output location (Kaggle input mounts are READ-ONLY) ─────────────────────
import sys
ON_KAGGLE = os.path.isdir('/kaggle/input')
if ON_KAGGLE:
    DOCS_DIR = '/kaggle/working'
    # Kaggle's mount layout has changed over time — attached datasets appear at
    # /kaggle/input/<slug> on older images and at
    # /kaggle/input/datasets/<owner>/<slug> on current ones — so discover the
    # directories that actually hold files instead of hardcoding a path.
    KAGGLE_INPUTS = sorted({r for r, _sub, f in os.walk('/kaggle/input') if f})
    # Link the read-only inputs into the writable dir, so that every existing
    # os.path.join(DATA_DIR | DOCS_DIR, ...) works for BOTH reads and writes.
    for _d in KAGGLE_INPUTS:
        if _d not in sys.path:
            sys.path.append(_d)
        for _f in os.listdir(_d):
            _s, _dst = os.path.join(_d, _f), os.path.join(DOCS_DIR, _f)
            if os.path.isfile(_s) and not os.path.exists(_dst):
                os.symlink(_s, _dst)
    print('Kaggle inputs discovered:', KAGGLE_INPUTS)
else:
    KAGGLE_INPUTS = []
    DOCS_DIR = "/Users/islamriajul/Documents"
from german_epf_research import (
    BSSM,
    DDNN,
    VIDDNN,
    ConformalWrapper,
    EvDNN,
    compute_all_metrics,
    inv_signed_log,
    signed_log,
)
import optuna
optuna.logging.set_verbosity(optuna.logging.WARNING)

GITHUB_STATIC = False
def _pick_renderer():
    if GITHUB_STATIC:
        return 'svg'
    if 'KAGGLE_KERNEL_RUN_TYPE' in os.environ:
        return 'kaggle'
    try:
        import google.colab
        return 'colab'
    except Exception:
        pass
    return 'notebook'
pio.renderers.default = _pick_renderer()

import sys
for _p in KAGGLE_INPUTS + ['.', '/Users/islamriajul/Documents']:
    if _p not in sys.path:
        sys.path.append(_p)

from german_epf_research import (
    DDNN, EvDNN, VIDDNN, BSSM,
    StudentTDNN_Adam, interval_metrics,
    signed_log, inv_signed_log, compute_all_metrics,
)

warnings.filterwarnings('ignore')
np.random.seed(42)

def data_dir():
    candidates = [
        '/kaggle/working' if ON_KAGGLE else 'data',
        'data',
        '/Users/islamriajul/Documents',
        '.',
    ]
    for d in candidates:
        if os.path.isdir(d):
            return d
    return '.'

DATA_DIR = data_dir()
def dpath(fname):
    p = os.path.join(DATA_DIR, fname)
    return p if os.path.exists(p) else fname

print(f'Setup complete. Data directory: {DATA_DIR}')