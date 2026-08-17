import os
import numpy as np
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from setup_imports import DATA_DIR

def load_if_exists(filename):
    paths = [os.path.join(DATA_DIR, f"{filename}"), filename]
    for p in paths:
        if os.path.exists(p):
            return np.load(p, allow_pickle=True).item()
    return None


def collect_rows():
    rows = []
    d8 = load_if_exists('results_part8.npy')
    if d8 is not None:
        if 'results_selected' in d8:
            for name, r in d8['results_selected'].items():
                rows.append(('Part 8 (Selected)', name, r.get('MAE'), r.get('PICP_90'),
                             r.get('MPIW_90'), r.get('MAACE')))
        if 'results_all' in d8:
            for name, r in d8['results_all'].items():
                rows.append(('Part 8 (All)', name, r.get('MAE'), r.get('PICP_90'),
                             r.get('MPIW_90'), r.get('MAACE')))
        if 'results' in d8:
            for name, r in d8['results'].items():
                rows.append(('Part 8', name, r.get('MAE'), r.get('PICP_90'),
                             r.get('MPIW_90'), r.get('MAACE')))

    d14 = load_if_exists('part14_results.npy')
    if d14 is not None:
        for name, m in d14['table'].items():
            rows.append(('Part 14', name, m.get('MAE'), m.get('PICP_90'),
                         m.get('MPIW_90_med', m.get('MPIW_90')), m.get('MAACE')))

    d15 = load_if_exists('part15_results.npy')
    if d15 is not None:
        for key, label in [('before', 'StudentT plain-CP (P15)'),
                           ('after', 'StudentT scale-aware (P15)')]:
            if key in d15:
                m = d15[key]
                rows.append(('Part 15', label, m.get('MAE'), m.get('PICP_90'),
                             m.get('MPIW_90_med', m.get('MPIW_90')), m.get('MAACE')))

    d15b = load_if_exists('part15b_results.npy')
    if d15b is not None:
        m = d15b['metrics']
        rows.append(('Part 15b', 'StudentT multi-level (FINAL)', m.get('MAE'),
                     m.get('PICP_90'), m.get('MPIW_90_med', m.get('MPIW_90')),
                     d15b.get('maace')))

    return rows


def print_table(rows):
    print("\n" + "="*95)
    print("  UNIFIED MODEL COMPARISON  —  same test window (May 2025 – Apr 2026)")
    print("="*95)
    print(f"  {'Source':<20}{'Model':<30}{'MAE':>8}{'PICP_90':>10}{'MPIW':>9}{'MAACE':>9}")
    print("  " + "-"*91)
    
    rows_sorted = sorted(rows, key=lambda r: (r[2] is None, r[2] if r[2] is not None else 1e9))
    best_mae = min((r[2] for r in rows if r[2] is not None), default=None)
    
    for src, name, mae, picp, mpiw, maace in rows_sorted:
        mae_s   = f"{mae:.2f}"   if mae   is not None else "  —"
        picp_s  = f"{picp:.1%}"  if picp  is not None else "  —"
        mpiw_s  = f"{mpiw:.1f}"  if mpiw  is not None else "  —"
        maace_s = f"{maace:.2f}%" if maace is not None else "  —"
        star = "  <= best MAE" if (mae is not None and mae == best_mae) else ""
        print(f"  {src:<20}{name:<30}{mae_s:>8}{picp_s:>10}{mpiw_s:>9}{maace_s:>9}{star}")
    
    print("="*95)


def plot_unified_plotly(rows, save_path=os.path.join(DATA_DIR, "part18_unified_comparison.html")):
    print("\nGenerating interactive Plotly unified comparison dashboard...")
    rows = [r for r in rows if r[2] is not None and r[5] is not None]
    rows_sorted = sorted(rows, key=lambda r: r[2])
    
    names = [r[1] for r in rows_sorted]
    sources = [r[0] for r in rows_sorted]
    mae = [r[2] for r in rows_sorted]
    maace = [r[5] for r in rows_sorted]
    picp = [(r[3]*100 if r[3] is not None else np.nan) for r in rows_sorted]

    srccol = {
        'Part 8': '#888780', 'Part 8 (Selected)': '#7F77DD', 'Part 8 (All)': '#BA7517',
        'Part 14': '#378ADD', 'Part 15': '#1D9E75', 'Part 15b': '#0C447C'
    }
    cols = [srccol.get(s, '#888780') for s in sources]

    fig = make_subplots(
        rows=1, cols=3,
        subplot_titles=(
            "<b>MAE (€/MWh) — lower better</b>",
            "<b>MAACE (%) — lower = better calibrated</b>",
            "<b>PICP 90% — target 90%</b>"
        ),
        horizontal_spacing=0.10,
        shared_yaxes=True
    )

    fig.add_trace(go.Bar(y=names, x=mae, orientation='h', marker_color=cols, text=[f"{v:.1f}" for v in mae], textposition='inside', showlegend=False, hovertemplate="<b>%{y}</b><br>MAE: €%{x:.2f}<extra></extra>"), row=1, col=1)
    fig.add_trace(go.Bar(y=names, x=maace, orientation='h', marker_color=cols, text=[f"{v:.1f}%" for v in maace], textposition='inside', showlegend=False, hovertemplate="<b>%{y}</b><br>MAACE: %{x:.2f}%<extra></extra>"), row=1, col=2)
    fig.add_trace(go.Bar(y=names, x=picp, orientation='h', marker_color=cols, text=[f"{v:.0f}%" for v in picp], textposition='inside', showlegend=False, hovertemplate="<b>%{y}</b><br>PICP: %{x:.1f}%<extra></extra>"), row=1, col=3)

    fig.add_shape(type="line", x0=90, x1=90, y0=-0.5, y1=len(names)-0.5, line=dict(color="#D85A30", dash="dash", width=2), row=1, col=3)

    fig.update_layout(
        title=dict(text="<b>Part 18 — All Models Compared on the Same Test Window</b>", font=dict(size=15), x=0.5, xanchor="center"),
        paper_bgcolor="#f7f6f3", plot_bgcolor="#ffffff",
        height=max(600, len(names) * 35), width=1500,
        yaxis=dict(autorange="reversed")
    )

    fig.update_xaxes(title_text="MAE (€/MWh)", row=1, col=1)
    fig.update_xaxes(title_text="MAACE Error (%)", row=1, col=2)
    fig.update_xaxes(title_text="PICP Coverage (%)", row=1, col=3)

    fig.write_html(save_path)
    print(f"  ✓ Saved interactive unified comparison dashboard to {save_path}")
    fig.show(renderer="iframe")


if __name__ == "__main__":
    rows = collect_rows()
    if not rows:
        print("No result files found. Run previous parts first.")
    else:
        found = sorted(set(r[0] for r in rows))
        print(f"\nFound results from sources: {', '.join(found)}")
        print_table(rows)
        plot_unified_plotly(rows)
        print("\n✓ Part 18 complete!")