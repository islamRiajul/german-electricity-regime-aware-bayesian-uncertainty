import os
import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestRegressor
from sklearn.feature_selection import mutual_info_regression
import plotly.graph_objects as go
from setup_imports import DATA_DIR

def per_regime_analysis(df):
    print("Analyzing feature importance per regime...")
    groups = {
        'Conventional':  ['residual_load', 'other_gen'],
        'Renewable':     ['wind_offshore', 'wind_onshore', 'solar', 'pv_wind'],
        'Demand':        ['load_forecast', 'total_gen'],
    }
    all_feats = [f for g in groups.values() for f in g if f in df.columns]
    names = {0: 'Normal', 1: 'Elevated', 2: 'Crisis'}
    group_share = {}

    for r in [0, 1, 2]:
        sub = df[df['regime'] == r]
        X = StandardScaler().fit_transform(sub[all_feats].fillna(0).values)
        y = sub['price'].values
        
        mi = mutual_info_regression(X, y, random_state=42)
        mi = mi / (mi.max() + 1e-9)
        rf = RandomForestRegressor(n_estimators=80, max_depth=7,
                                   min_samples_leaf=30, random_state=42, n_jobs=-1)
        rf.fit(X, y)
        rfi = rf.feature_importances_ / (rf.feature_importances_.max() + 1e-9)
        imp = dict(zip(all_feats, (mi + rfi) / 2))

        gs = {g: sum(imp[f] for f in fl if f in imp) for g, fl in groups.items()}
        tot = sum(gs.values())
        group_share[r] = {g: v / tot * 100 for g, v in gs.items()}

        print(f"\n  {names[r]} regime (avg €{y.mean():.0f}):")
        print(f"    Conventional  : {group_share[r]['Conventional']:5.1f}%")
        print(f"    Renewable     : {group_share[r]['Renewable']:5.1f}%")
        print(f"    Demand        : {group_share[r]['Demand']:5.1f}%")

    return group_share, names


def visualize_per_regime_plotly(group_share, names, save_path=os.path.join(DATA_DIR, "part11_per_regime.html")):
    print("\nVisualizing per-regime importance with Plotly...")
    regimes = [0, 1, 2]
    
    conv = [group_share[r]['Conventional'] for r in regimes]
    ren  = [group_share[r]['Renewable'] for r in regimes]
    dem  = [group_share[r]['Demand'] for r in regimes]

    regime_labels = [f"{names[r]}<br>(avg €{[64, 111, 236][r]})" for r in regimes]

    fig = go.Figure()
    fig.add_trace(go.Bar(x=regime_labels, y=conv, name='Conventional (gas/coal need)', marker_color='#5F5E5A', hovertemplate="<b>%{x}</b><br>Conventional Share: %{y:.1f}%<extra></extra>"))
    fig.add_trace(go.Bar(x=regime_labels, y=ren, name='Renewable (wind/solar)', marker_color='#639922', hovertemplate="<b>%{x}</b><br>Renewable Share: %{y:.1f}%<extra></extra>"))
    fig.add_trace(go.Bar(x=regime_labels, y=dem, name='Demand', marker_color='#85B7EB', hovertemplate="<b>%{x}</b><br>Demand Share: %{y:.1f}%<extra></extra>"))

    fig.update_layout(
        barmode='stack',
        title=dict(
            text=f"<b>What drives price changes by regime</b><br><sup>Renewables jump from {ren[0]:.0f}% (normal) to {ren[2]:.0f}% (crisis)</sup>",
            font=dict(size=16), x=0.5, xanchor="center"
        ),
        xaxis_title="<b>Market Regime & Average Price</b>",
        yaxis_title="<b>Share of feature importance (%)</b>",
        yaxis=dict(range=[0, 100]),
        paper_bgcolor="#f7f6f3", plot_bgcolor="#ffffff",
        height=750, width=1100,
        legend=dict(orientation="h", yanchor="bottom", y=-0.25, xanchor="center", x=0.5)
    )

    fig.update_xaxes(showgrid=False)
    fig.update_yaxes(showgrid=True, gridwidth=1, gridcolor="rgba(0,0,0,0.1)")

    fig.write_html(save_path)
    print(f"  ✓ Saved interactive report to {save_path}")
    fig.show()


if __name__ == "__main__":
    df = pd.read_pickle(os.path.join(DATA_DIR, "data_part7_regimes.pkl"))
    group_share, names = per_regime_analysis(df)
    visualize_per_regime_plotly(group_share, names, os.path.join(DATA_DIR, "part11_per_regime.html"))
    print("\n✓ Done — interactive dashboard saved!")