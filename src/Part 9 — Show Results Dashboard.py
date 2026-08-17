import os
import numpy as np
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from setup_imports import DATA_DIR, DOCS_DIR

def show_results_tables(results_selected, results_all):
    print("\n" + "="*84)
    print("  RESULTS TABLE — SELECTED FEATURES TEST SET (May 2025 – End)")
    print("="*84)
    print(f"  {'Model':<12}{'MAE':>9}{'RMSE':>9}{'CRPS':>9}"
          f"{'PICP_90':>10}{'MPIW_90':>10}{'MAACE':>9}")
    print("  " + "-"*76)
    for name, m in results_selected.items():
        print(f"  {name:<12}{m['MAE']:>9.2f}{m['RMSE']:>9.2f}{m['CRPS']:>9.2f}"
              f"{m['PICP_90']:>9.1%}{m['MPIW_90']:>10.2f}{m['MAACE']:>8.2f}%")
    print("="*84)

    print("\n" + "="*84)
    print("  RESULTS TABLE — ALL CANDIDATE FEATURES TEST SET (May 2025 – End)")
    print("="*84)
    print(f"  {'Model':<12}{'MAE':>9}{'RMSE':>9}{'CRPS':>9}"
          f"{'PICP_90':>10}{'MPIW_90':>10}{'MAACE':>9}")
    print("  " + "-"*76)
    for name, m in results_all.items():
        print(f"  {name:<12}{m['MAE']:>9.2f}{m['RMSE']:>9.2f}{m['CRPS']:>9.2f}"
              f"{m['PICP_90']:>9.1%}{m['MPIW_90']:>10.2f}{m['MAACE']:>8.2f}%")
    print("="*84)


def plot_predictions_plotly(results_selected, results_all, y_test, test_dates=None, save_path=os.path.join(DOCS_DIR, 'part9_predictions.html')):
    print("\nGenerating high-performance WebGL Plotly dashboard for both feature sets...")
    
    combined_results = {}
    for name, m in results_selected.items():
        combined_results[f"{name} (Selected Features)"] = m
    for name, m in results_all.items():
        combined_results[f"{name} (All Candidate Features)"] = m

    models = list(combined_results.keys())
    n_show = len(y_test)

    colors = {
        'DDNN (Selected Features)': '#378ADD',
        'EvDNN (Selected Features)': '#D85A30',
        'VI-DDNN (Selected Features)': '#1D9E75',
        'VI+CP (Selected Features)': '#7F77DD',
        'BSSM (Selected Features)': '#BA7517',
        'DDNN (All Candidate Features)': '#1B4F72',
        'EvDNN (All Candidate Features)': '#B03A2E',
        'VI-DDNN (All Candidate Features)': '#117A65',
        'VI+CP (All Candidate Features)': '#512E5F',
        'BSSM (All Candidate Features)': '#7E5109'
    }

    x_vals = test_dates[:n_show] if test_dates is not None else list(range(n_show))

    subplot_titles = [
        f"<b>{name}</b> — MAE: €{m['MAE']:.1f} | CRPS: {m['CRPS']:.2f} | MAACE: {m['MAACE']:.1f}%"
        for name, m in combined_results.items()
    ]
    
    fig = make_subplots(
        rows=len(models),
        cols=1,
        shared_xaxes=True,
        vertical_spacing=0.03,
        subplot_titles=subplot_titles
    )

    for i, name in enumerate(models, start=1):
        m = combined_results[name]
        mu = m['mu'][:n_show]
        
        if 'lower' in m and 'upper' in m:
            lo = m['lower'][:n_show]
            hi = m['upper'][:n_show]
        else:
            sig = m.get('sigma', np.abs(m['mu'] - y_test[:n_show]))[:n_show]
            lo = mu - 1.645 * sig
            hi = mu + 1.645 * sig

        c = colors.get(name, '#378ADD')

        rgb_tuple = tuple(int(c.lstrip('#')[j:j+2], 16) for j in (0, 2, 4))
        fill_color = f"rgba({rgb_tuple[0]}, {rgb_tuple[1]}, {rgb_tuple[2]}, 0.22)"

        fig.add_trace(
            go.Scattergl(
                x=x_vals, y=hi,
                mode='lines', line=dict(width=0),
                showlegend=False, hoverinfo='skip'
            ),
            row=i, col=1
        )

        fig.add_trace(
            go.Scattergl(
                x=x_vals, y=lo,
                mode='lines', line=dict(width=0),
                fill='tonexty', fillcolor=fill_color,
                name='90% Band', showlegend=(i == 1)
            ),
            row=i, col=1
        )

        fig.add_trace(
            go.Scattergl(
                x=x_vals, y=y_test[:n_show],
                mode='lines', line=dict(color='#222222', width=1.1),
                name='Actual Price', showlegend=(i == 1)
            ),
            row=i, col=1
        )

        fig.add_trace(
            go.Scattergl(
                x=x_vals, y=mu,
                mode='lines', line=dict(color=c, width=1.8),
                name=f'{name} Forecast', showlegend=False
            ),
            row=i, col=1
        )

        fig.update_yaxes(title_text="€/MWh", row=i, col=1, gridcolor='#E8E8E8')

    fig.update_layout(
        height=280 * len(models),
        width=1300,
        title_text="<b>Model Predictions vs. Actual Prices: Selected vs. All Candidate Features</b>",
        title_x=0.5,
        template="plotly_white",
        legend=dict(orientation="h", yanchor="bottom", y=1.01, xanchor="right", x=1),
        hovermode="x unified"
    )

    fig.update_xaxes(gridcolor='#E8E8E8')
    fig.update_xaxes(title_text="Time / Hour", row=len(models), col=1)

    fig.write_html(save_path)
    print(f"  ✓ Saved interactive Plotly dashboard to -> {save_path}")
    
    try:
        fig.show()
    except Exception:
        print(f"  (Note: Open '{save_path}' directly in your web browser to view)")


if __name__ == "__main__":
    results_path = os.path.join(DOCS_DIR, 'results_part8.npy')
    if not os.path.exists(results_path):
        results_path = 'results_part8.npy'
        
    data = np.load(results_path, allow_pickle=True).item()
    results_selected = data.get('results_selected', data.get('results', {}))
    results_all = data.get('results_all', {})
    y_test = data['y_test']
    test_dates = data.get('test_dates', None)

    show_results_tables(results_selected, results_all)
    plot_predictions_plotly(results_selected, results_all, y_test, test_dates)
    print("\n✓ Done — view dashboard above or open part9_predictions.html in your browser")