# Making the Plots Visible on GitHub

This document is derived from what the code in this repo actually does, not from
what would be convenient. Read the "Known gaps" section before trusting the
committed `figures/` directory to regenerate itself.

## What the code actually produces

Every figure in this project is a Plotly figure written with a single call:

```python
fig.write_html(save_path)
```

There are 15 such calls, in `src/Part 3, 5, 6, 7, 9, 10, 11, 12 (×2), 13, 14 (×2),
15, 15b, 16, 18` and `src/german_epf_research.py:1111`. The remaining HTML files
in `figures/` (the `part17_*`, `part19_*`, and `*_all_models_*` ones) are written
by notebook cells that follow the same pattern.

Two consequences follow directly from that call:

- **`write_html` defaults to `include_plotlyjs=True`**, so each file embeds a
  complete copy of plotly.js. That is why nearly every HTML file is ~4.4 MB
  regardless of how much data it plots.
- **No PNG is ever written by this code.** There is no `write_image` or
  `to_image` call anywhere in `src/` or in the notebook — `grep` finds zero.
  `kaleido>=1.0` is listed in `requirements.txt`, but nothing calls it.

## What renders where

- **Static `.png` images** render inline in the GitHub file viewer and can be
  embedded in Markdown:

  ```markdown
  ![Data overview](figures/part3_data_overview.png)
  ```

  The 12 PNGs currently in `figures/` do render this way. See "Known gaps" for
  why you cannot regenerate them.

- **Interactive `.html` files** do **not** render on github.com — GitHub serves
  them as raw text or as a download. Three ways to view them:

  1. **Download** the `.html` and open it locally.
  2. **GitHub Pages** — enable Pages on the repo, then each file is live at
     `https://islamriajul.github.io/german-electricity-regime-aware-bayesian-uncertainty/figures/part3_data_overview.html`
  3. **htmlpreview** — prefix the blob URL:
     `https://htmlpreview.github.io/?https://github.com/islamRiajul/german-electricity-regime-aware-bayesian-uncertainty/blob/main/figures/part3_data_overview.html`
     (Note: htmlpreview struggles with multi-MB files; Pages is more reliable here.)

- **The notebook.** GitHub renders committed notebook outputs, but a Plotly
  figure using the default `notebook`/`iframe` renderer stores nothing GitHub can
  draw. `src/Setup & Imports.py:39` exists for exactly this:

  ```python
  GITHUB_STATIC = False   # -> renderer 'svg' when True
  ```

  Set it to `True` and re-run all cells before committing, and each figure is
  baked into the notebook as a static SVG that shows up on the repo page. It is
  currently `False` in both `src/Setup & Imports.py:39` and the corresponding
  notebook cell, so the committed notebook has no visible plots.

  Also note `.gitignore` excludes `German_Electricity_Uncertainity.executed.ipynb`
  — the output of `run_all.py` — so the executed copy never reaches GitHub even if
  it does contain rendered figures.

## Known gaps

These are real mismatches between the code and the committed `figures/`
directory. A fresh clone will not reproduce what is checked in.

**1. Output paths are hardcoded to a personal directory.**
`src/Setup & Imports.py:25` sets

```python
DOCS_DIR = "/Users/islamriajul/Documents"
```

and most save paths are `os.path.join(DOCS_DIR, "partN_....html")`. On any other
machine that directory does not exist and the write fails. `src/german_epf_research.py:1280`
hardcodes `/Users/islamriajul/Documents/epf_results_plotly.html` the same way.

**2. Nothing writes to `figures/`.** The figures that do resolve locally go to
`DOCS_DIR`, and the ones built on `DATA_DIR` (Parts 11, 15, 15b, 16, 18) go to
`data/` — because `data_dir()` in `src/Setup & Imports.py:67` returns `data` when
run from the repo root. The committed `figures/` directory was assembled by hand.

**3. The PNGs cannot be regenerated.** Since no code calls `write_image`, the 12
PNGs were produced outside this pipeline. Re-running everything gives you 30 HTML
files and zero PNGs — including for `part5_feature_selection.png`,
`part6_signed_log.png`, and `part7_regime_detection.png`, which have no HTML
counterpart on disk at all.

**4. Some code defaults do not match the committed filenames.**

| Code default | On disk |
|---|---|
| `part7_regimes.html` (`Part 7:66`) | only `part7_regime_detection.png` |
| `part6_transform.html` (`Part 6:25`) | only `part6_signed_log.png` |
| `part4_feature_selection.html` (written by `Part 5:7`) | `part4_feature_selection.html` + `part5_feature_selection.png` |

**5. The notebook links to figures that do not exist.** It references
`figures/figure_9.html`, `figure_17.html`, … `figure_60.html` — an older naming
scheme. None of those files are in `figures/`, so those links are dead.

## Fixing it

A single project-relative figures directory fixes gaps 1–2. In
`src/Setup & Imports.py`, replace the hardcoded constant:

```python
import os
REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
FIG_DIR   = os.environ.get("FIG_DIR", os.path.join(REPO_ROOT, "figures"))
os.makedirs(FIG_DIR, exist_ok=True)
DOCS_DIR  = FIG_DIR          # keeps every existing save_path working
```

Then point the `DATA_DIR`-based save paths (Parts 11, 15, 15b, 16, 18) at
`FIG_DIR` too, so data and figures stop sharing a directory.

To get PNGs alongside the HTML — gap 3 — add the export next to each
`write_html`. `kaleido` is already a dependency, so a small helper used in place
of `fig.write_html(save_path)` is enough:

```python
def save_fig(fig, save_path, png=True, cdn=True):
    fig.write_html(save_path, include_plotlyjs="cdn" if cdn else True)
    if png:
        fig.write_image(save_path.replace(".html", ".png"), scale=2)
```

## Note on file size

The interactive `.html` files embed the full plotly.js bundle, which is why
`figures/` is ~161 MB of this repo's ~226 MB. Passing
`include_plotlyjs="cdn"` — as the helper above does — drops each file to tens of
kilobytes and would bring the repo to roughly 70 MB, at the cost of the figures
needing a network connection to render. Keeping only the PNGs in git and
regenerating HTML on demand via `run_all.py` is the other option.
