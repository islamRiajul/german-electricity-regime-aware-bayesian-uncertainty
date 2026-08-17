import os
import numpy as np
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from setup_imports import DATA_DIR, DOCS_DIR

def analyze_results_plotly(results_selected, results_all, save_path=os.path.join(DOCS_DIR, "part10_analysis.html")):
    print("Analyzing results for 16 and 35 features with Plotly...")
    models = ['DDNN', 'EvDNN', 'VI-DDNN', 'VI+CP', 'BSSM']
    
    colors = {
        'DDNN': '#378ADD',
        'EvDNN': '#D85A30',
        'VI-DDNN': '#1D9E75',
        'VI+CP': '#7F77DD',
        'BSSM': '#BA7517'
    }

    fig = make_subplots(
        rows=2, cols=3,
        subplot_titles=(
            "1. MAE (€/MWh) — Accuracy",
            "2. CRPS — Distribution Quality",
            "3. MAACE (%) — Calibration Error",
            "4. Calibration Curve (Selected vs. All Candidate Features)"
        ),
        specs=[
            [{"type": "xy"}, {"type": "xy"}, {"type": "xy"}],
            [{"colspan": 3, "type": "xy"}, None, None]
        ],
        vertical_spacing=0.18,
        horizontal_spacing=0.08
    )

    metrics_config = [
        ('MAE', 1, 1, 'MAE (€/MWh)', True),
        ('CRPS', 1, 2, 'CRPS', True),
        ('MAACE', 1, 3, 'MAACE (%)', True)
    ]

    for metric, row, col, label, lower_better in metrics_config:
        vals_sel = [results_selected[m][metric] for m in models if m in results_selected]
        fig.add_trace(
            go.Bar(
                x=models, y=vals_sel,
                name='Selected (16 Features)',
                marker_color=[colors.get(m, '#888') for m in models],
                text=[f"{v:.1f}" for v in vals_sel],
                textposition='outside', legendgroup='sel',
                showlegend=(row == 1 and col == 1),
                hovertemplate="Model: %{x}<br>Selected: %{y:.2f}<extra></extra>"
            ),
            row=row, col=col
        )

        vals_all = [results_all[m][metric] for m in models if m in results_all]
        fig.add_trace(
            go.Bar(
                x=models, y=vals_all,
                name='All Candidate (35 Features)',
                marker_color=[colors.get(m, '#888') for m in models],
                marker_pattern_shape="x",
                text=[f"{v:.1f}" for v in vals_all],
                textposition='outside', legendgroup='all',
                showlegend=(row == 1 and col == 1),
                hovertemplate="Model: %{x}<br>All Candidate: %{y:.2f}<extra></extra>"
            ),
            row=row, col=col
        )
        
        fig.update_xaxes(title_text="", row=row, col=col)
        fig.update_yaxes(title_text=label, row=row, col=col)

    fig.update_layout(barmode='group')

    feature_sets = [
        ('Selected (16 Features)', results_selected, 'solid'),
        ('All Candidate (35 Features)', results_all, 'dash')
    ]
    
    for fs_label, res_dict, dash_style in feature_sets:
        for name in models:
            if name not in res_dict:
                continue
            m = res_dict[name]
            levels = np.array(m['levels']) * 100
            picps = np.array(m['picps']) * 100
            col = colors.get(name, '#888')

            fig.add_trace(
                go.Scatter(
                    x=levels, y=picps,
                    mode='lines+markers',
                    line=dict(color=col, width=2, dash=dash_style),
                    marker=dict(size=5),
                    name=f"{name} ({fs_label})",
                    hovertemplate=f"<b>%{{fullData.name}}</b><br>Claimed Confidence: %{{x}}%<br>Actual Coverage: %{{y:.1f}}%<extra></extra>"
                ),
                row=2, col=1
            )

    fig.add_trace(
        go.Scatter(
            x=[10, 90], y=[10, 90],
            mode='lines', line=dict(color="black", dash="dash", width=1.5),
            name="Perfect calibration", hoverinfo="skip"
        ),
        row=2, col=1
    )

    fig.add_trace(
        go.Scatter(
            x=[10, 90, 90, 10], y=[5, 85, 95, 15],
            fill='toself', fillcolor='rgba(128, 128, 128, 0.08)',
            line=dict(color='rgba(255,255,255,0)'),
            name="Acceptable Zone", hoverinfo="skip"
        ),
        row=2, col=1
    )

    fig.update_layout(
        title=dict(
            text="<b>Result Analysis — Accuracy and Uncertainty Calibration Dashboard (16 vs 35 Features)</b>",
            font=dict(size=16), x=0.5, xanchor="center", y=0.96
        ),
        paper_bgcolor="#f7f6f3", plot_bgcolor="#ffffff",
        height=1050, width=1350,
        margin=dict(t=220, l=60, r=60, b=60),
        legend=dict(orientation="h", yanchor="bottom", y=1.06, xanchor="center", x=0.5, font=dict(size=10))
    )

    fig.update_xaxes(title_text="Confidence the model claims (%)", row=2, col=1)
    fig.update_yaxes(title_text="How often the truth was actually in the band (%)", row=2, col=1)
    fig.update_xaxes(showgrid=True, gridwidth=1, gridcolor="rgba(0,0,0,0.1)")
    fig.update_yaxes(showgrid=True, gridwidth=1, gridcolor="rgba(0,0,0,0.1)")

    fig.write_html(save_path)
    print(f"  ✓ Saved interactive report to {save_path}")
    fig.show()


def print_analysis(results_selected, results_all):
    print("\n  PLAIN-LANGUAGE ANALYSIS (16 vs 35 Features):")
    best_mae_sel = min(results_selected, key=lambda m: results_selected[m]['MAE'])
    best_mae_all = min(results_all, key=lambda m: results_all[m]['MAE'])
    print(f"  • Best MAE (Selected 16): {best_mae_sel} at €{results_selected[best_mae_sel]['MAE']:.1f}")
    print(f"  • Best MAE (All 35): {best_mae_all} at €{results_all[best_mae_all]['MAE']:.1f}")
    
    best_cal_sel = min(results_selected, key=lambda m: results_selected[m]['MAACE'])
    best_cal_all = min(results_all, key=lambda m: results_all[m]['MAACE'])
    print(f"  • Best Calibration MAACE (Selected 16): {best_cal_sel} at {results_selected[best_cal_sel]['MAACE']:.1f}%")
    print(f"  • Best Calibration MAACE (All 35): {best_cal_all} at {results_all[best_cal_all]['MAACE']:.1f}%")


if __name__ == "__main__":
    data = np.load(os.path.join(DATA_DIR, "results_part8.npy"), allow_pickle=True).item()
    results_selected = data.get('results_selected', data.get('results', {}))
    results_all = data.get('results_all', {})
    
    analyze_results_plotly(results_selected, results_all, os.path.join(DATA_DIR, "part10_analysis.html"))
    print_analysis(results_selected, results_all)
    print("\n✓ Done — interactive dashboard saved!")