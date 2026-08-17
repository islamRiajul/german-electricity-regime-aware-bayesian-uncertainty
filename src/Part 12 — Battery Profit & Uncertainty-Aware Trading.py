import os
import numpy as np
import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from setup_imports import DATA_DIR, DOCS_DIR

def run_battery(pred, regime_of_hour, eta=0.9):
    mu, sigma, y_true = pred['mu'], pred['sigma'], pred['y_true']
    n_days = len(mu) // 24

    prof_A = {0: 0.0, 1: 0.0, 2: 0.0}
    prof_B = {0: 0.0, 1: 0.0, 2: 0.0}
    days   = {0: 0, 1: 0, 2: 0}

    for d in range(n_days):
        sl = slice(d * 24, (d + 1) * 24)
        mu_d, sig_d, true_d = mu[sl], sigma[sl], y_true[sl]
        reg_d = regime_of_hour[sl]
        if len(mu_d) < 24:
            continue
        regime = int(np.bincount(reg_d).argmax())
        days[regime] += 1

        buy_h = int(np.argmin(mu_d))
        sell_h = int(np.argmax(mu_d))
        if sell_h <= buy_h:
            buy_h, sell_h = min(buy_h, sell_h), max(buy_h, sell_h)

        realised = true_d[sell_h] * eta - true_d[buy_h]
        prof_A[regime] += realised

        gap = mu_d[sell_h] - mu_d[buy_h]
        doubt = 1.28 * (sig_d[buy_h] + sig_d[sell_h])
        if gap > doubt:
            prof_B[regime] += realised

    names = {0: 'Normal', 1: 'Elevated', 2: 'Crisis'}
    rows = []
    for r in [0, 1, 2]:
        dd = max(days[r], 1)
        a = prof_A[r] / dd
        b = prof_B[r] / dd
        rows.append((names[r], a, b, days[r]))
        print(f"  {names[r]:9s}: Trader A €{a:7.2f}/day  Trader B €{b:7.2f}/day  (extra €{b-a:6.2f})  [{days[r]} days]")
    return rows


def visualize_battery_plotly(rows, save_path=os.path.join(DOCS_DIR, "part12_battery_profit.html")):
    print("\nVisualizing battery profit with Plotly...")
    regs = [r[0] for r in rows]
    A = [r[1] for r in rows]
    B = [r[2] for r in rows]
    rcol = {'Normal': '#1D9E75', 'Elevated': '#BA7517', 'Crisis': '#D85A30'}

    fig = make_subplots(
        rows=1, cols=2,
        subplot_titles=(
            "<b>1. Daily Profit Comparison per MWh</b><br><sup>Uncertainty-aware trading wins most in crisis</sup>",
            "<b>2. Annual Value Unlocked Across Germany's 2030 Fleet</b><br><sup>Target: 10 GW / 10 GWh battery capacity</sup>"
        ),
        horizontal_spacing=0.12
    )

    fig.add_trace(
        go.Bar(
            x=regs, y=A, name='Trader A (always bet)', marker_color='#888780',
            text=[f"€{val:.0f}" for val in A], textposition='outside',
            hovertemplate="<b>%{x}</b><br>Trader A Profit: €%{y:.2f}/day<extra></extra>"
        ),
        row=1, col=1
    )

    fig.add_trace(
        go.Bar(
            x=regs, y=B, name='Trader B (bet when sure)', marker_color=[rcol[r] for r in regs],
            text=[f"€{val:.0f}" for val in B], textposition='outside',
            hovertemplate="<b>%{x}</b><br>Trader B Profit: €%{y:.2f}/day<extra></extra>"
        ),
        row=1, col=1
    )

    fleet = 10000
    fleet_val = [(b - a) * 365 * fleet / 1e6 for _, a, b, _ in rows]

    fig.add_trace(
        go.Bar(
            x=regs, y=fleet_val, marker_color=[rcol[r] for r in regs],
            name='Fleet Value (€M)', showlegend=False,
            text=[f"€{val:.0f}M" for val in fleet_val], textposition='outside',
            hovertemplate="<b>%{x} Regime</b><br>Extra Value: €%{y:.1f}M/year<extra></extra>"
        ),
        row=1, col=2
    )

    fig.update_layout(
        barmode='group',
        title=dict(
            text="<b>Battery Profit Story — Uncertainty Quantification Is Worth Most When the Market Is Volatile</b>",
            font=dict(size=16), x=0.5, xanchor="center"
        ),
        paper_bgcolor="#f7f6f3", plot_bgcolor="#ffffff",
        height=750, width=1350,
        legend=dict(orientation="h", yanchor="bottom", y=1.15, xanchor="center", x=0.5)
    )

    fig.update_yaxes(title_text="Battery profit (€ per MWh per day)", row=1, col=1)
    fig.update_yaxes(title_text="Extra value per year (€ million)", row=1, col=2)
    fig.update_xaxes(showgrid=False)
    fig.update_yaxes(showgrid=True, gridwidth=1, gridcolor="rgba(0,0,0,0.1)", range=[0, max(max(B)*1.15, max(fleet_val)*1.15)])

    fig.write_html(save_path)
    print(f"  ✓ Saved interactive report to {save_path}")
    try:
        fig.show()
    except Exception:
        pass


def make_full_predictions():
    df = pd.read_pickle(os.path.join(DATA_DIR, "data_part7_regimes.pkl"))
    price = df['price'].values
    mu = df['lag_24'].values
    resid = np.abs(price - mu)
    sigma = pd.Series(resid).rolling(168, min_periods=24).mean().bfill().values
    sigma = np.clip(sigma, 5, 200)

    valid = ~np.isnan(mu)
    return {
        'mu': mu[valid], 'sigma': sigma[valid], 'y_true': price[valid],
        'regime': df['regime'].values[valid], 'datetime': df['datetime'].values[valid]
    }


def evaluate_model_battery(mu, sigma, y_true, regimes, eta=0.9):
    n_days = len(mu) // 24
    total_A, total_B, n = 0.0, 0.0, 0
    for d in range(n_days):
        sl = slice(d * 24, (d + 1) * 24)
        mu_d, sig_d, true_d = mu[sl], sigma[sl], y_true[sl]
        if len(mu_d) < 24:
            continue
        n += 1
        buy_h = int(np.argmin(mu_d))
        sell_h = int(np.argmax(mu_d))
        if sell_h <= buy_h:
            buy_h, sell_h = min(buy_h, sell_h), max(buy_h, sell_h)
        realised = true_d[sell_h] * eta - true_d[buy_h]
        total_A += realised
        gap = mu_d[sell_h] - mu_d[buy_h]
        doubt = 1.28 * (sig_d[buy_h] + sig_d[sell_h])
        if gap > doubt:
            total_B += realised
    dd = max(n, 1)
    return total_A / dd, total_B / dd


def run_all_models_battery():
    data_8 = np.load(os.path.join(DOCS_DIR, 'results_part8.npy'), allow_pickle=True).item()
    results = data_8.get('results_selected', data_8.get('results', {}))
    y_true = data_8['y_test']

    df = pd.read_pickle(os.path.join(DOCS_DIR, 'data_part7_regimes.pkl'))
    test_df = df[df['datetime'] >= '2025-05-01'].copy()
    regimes = test_df['regime'].values

    model_performances = {}
    print("\n  Battery Arbitrage Performance Across All Models (Test Set):")
    print("  " + "─"*55)

    for name, res in results.items():
        mu = res['mu']
        sigma = res['sigma']
        min_len = min(len(mu), len(y_true))
        pa, pb = evaluate_model_battery(mu[:min_len], sigma[:min_len], y_true[:min_len], regimes[:min_len])
        model_performances[name] = {'Trader_A': pa, 'Trader_B': pb, 'Extra': pb - pa}
        print(f"  {name:<10}: Trader A: €{pa:.2f}/day | Trader B: €{pb:.2f}/day | Extra: €{pb-pa:.2f}/day")

    return model_performances


def visualize_all_models_plotly(performances, save_path=os.path.join(DOCS_DIR, "part12_all_models_battery.html")):
    print("\nVisualizing multi-model battery performance with Plotly...")
    models = list(performances.keys())
    trader_a = [performances[m]['Trader_A'] for m in models]
    trader_b = [performances[m]['Trader_B'] for m in models]
    fleet = 10000
    annual_extra_val = [(performances[m]['Extra']) * 365 * fleet / 1e6 for m in models]

    fig = make_subplots(
        rows=1, cols=2,
        subplot_titles=(
            "<b>1. Daily Profit per MWh (Trader A vs Trader B)</b>",
            "<b>2. Annual Extra Value Unlocked across 2030 Fleet (€M/year)</b>"
        ),
        horizontal_spacing=0.15
    )

    fig.add_trace(go.Bar(x=models, y=trader_a, name='Trader A (Always Bet)', marker_color='#888780', text=[f"€{v:.0f}" for v in trader_a], textposition='outside'), row=1, col=1)
    fig.add_trace(go.Bar(x=models, y=trader_b, name='Trader B (Uncertainty-Aware)', marker_color='#1D9E75', text=[f"€{v:.0f}" for v in trader_b], textposition='outside'), row=1, col=1)
    fig.add_trace(go.Bar(x=models, y=annual_extra_val, marker_color='#D85A30', name='Extra Fleet Value (€M)', showlegend=False, text=[f"€{v:.1f}M" for v in annual_extra_val], textposition='outside'), row=1, col=2)

    fig.update_layout(
        barmode='group',
        title=dict(text="<b>Multi-Model Battery Profit & Uncertainty Valuation (Test Set)</b>", font=dict(size=16), x=0.5, xanchor="center"),
        paper_bgcolor="#f7f6f3", plot_bgcolor="#ffffff",
        height=700, width=1350,
        legend=dict(orientation="h", yanchor="bottom", y=1.12, xanchor="center", x=0.5)
    )

    fig.update_yaxes(title_text="Daily Profit (€ / MWh / day)", row=1, col=1)
    fig.update_yaxes(title_text="Extra Value Generated (€ Million / year)", row=1, col=2)
    fig.update_xaxes(showgrid=False)
    fig.update_yaxes(showgrid=True, gridwidth=1, gridcolor="rgba(0,0,0,0.1)")

    fig.write_html(save_path)
    print(f"  ✓ Saved multi-model dashboard to {save_path}")
    try:
        fig.show()
    except Exception:
        pass


if __name__ == "__main__":
    print("Running battery simulation on full history (all regimes present)...\n")
    pred = make_full_predictions()
    regime_of_hour = pred['regime']

    rows = run_battery(pred, regime_of_hour)
    visualize_battery_plotly(rows, os.path.join(DOCS_DIR, "part12_battery_profit.html"))
    
    print("\nRunning battery simulation across all trained models from Part 8...\n")
    performances = run_all_models_battery()
    visualize_all_models_plotly(performances, os.path.join(DOCS_DIR, "part12_all_models_battery.html"))
    print("\n✓ Done — multi-model battery dashboards generated successfully!")