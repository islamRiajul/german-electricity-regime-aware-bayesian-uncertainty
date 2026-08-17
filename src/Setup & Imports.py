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
# -- Paths -------------------------------------------------------------------
# One writable working directory holds every generated file (.pkl / .npy /
# .json and the Plotly .html dashboards), because this pipeline reads and
# writes through the same DATA_DIR / DOCS_DIR names. Read-only sources are
# linked into it, so os.path.join(DATA_DIR, ...) resolves for BOTH.
#
#   Kaggle : /kaggle/working, with the attached datasets linked in
#   local  : $EPF_OUT, else <repo>/outputs, with data/ files/ figures/ src/ linked in
import sys

ON_KAGGLE = os.path.isdir('/kaggle/input')

def _repo_root():
    here = os.path.abspath(os.getcwd())
    while True:
        if os.path.isdir(os.path.join(here, '.git')):
            return here
        up = os.path.dirname(here)
        if up == here:
            return os.path.abspath(os.getcwd())
        here = up

if ON_KAGGLE:
    WORK_DIR = '/kaggle/working'
    # Kaggle's mount layout has changed over time: attached datasets appear at
    # /kaggle/input/<slug> on older images and /kaggle/input/datasets/<owner>/<slug>
    # on current ones, so discover the dirs that hold files instead of guessing.
    SOURCE_DIRS = sorted({r for r, _sub, f in os.walk('/kaggle/input') if f})
else:
    _ROOT = _repo_root()
    WORK_DIR = os.environ.get('EPF_OUT') or os.path.join(_ROOT, 'outputs')
    SOURCE_DIRS = [os.path.join(_ROOT, d) for d in ('data', 'files', 'figures', 'src')]
    SOURCE_DIRS = [d for d in SOURCE_DIRS if os.path.isdir(d)]

os.makedirs(WORK_DIR, exist_ok=True)
for _d in SOURCE_DIRS:
    if _d not in sys.path:
        sys.path.append(_d)
    for _f in os.listdir(_d):
        _s, _dst = os.path.join(_d, _f), os.path.join(WORK_DIR, _f)
        if os.path.isfile(_s) and not os.path.exists(_dst):
            os.symlink(_s, _dst)

DATA_DIR = DOCS_DIR = WORK_DIR
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
for _p in SOURCE_DIRS + ['.']:
    if _p not in sys.path:
        sys.path.append(_p)

from german_epf_research import (
    DDNN, EvDNN, VIDDNN, BSSM,
    StudentTDNN_Adam, interval_metrics,
    signed_log, inv_signed_log, compute_all_metrics,
)

warnings.filterwarnings('ignore')
np.random.seed(42)

# -- Data location -----------------------------------------------------------
def data_dir():
    """Everything lives in the one working directory (see Paths above)."""
    return DATA_DIR

def dpath(fname):
    """Build a path to a data file inside the working directory."""
    p = os.path.join(DATA_DIR, fname)
    return p if os.path.exists(p) else fname

print(f'Setup complete. Working directory: {WORK_DIR}')
print(f'  sources linked in: {SOURCE_DIRS}')
