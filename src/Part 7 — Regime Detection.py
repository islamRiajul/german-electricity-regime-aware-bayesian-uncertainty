import os
import numpy as np
import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from setup_imports import DATA_DIR, DOCS_DIR

def detect_regimes(df):
    print("Detecting market regimes...")
    price = df["price"].values
    dates = df["datetime"]

    roll_level = (
        pd.Series(price)
        .rolling(720, min_periods=48)
        .mean()
        .bfill()
        .values
    )
    roll_vol = (
        pd.Series(price)
        .rolling(168, min_periods=24)
        .std()
        .bfill()
        .values
    )

    tr_mask = dates < "2024-01-01"
    min_lvl, max_lvl = roll_level[tr_mask].min(), roll_level[tr_mask].max()
    min_vol, max_vol = roll_vol[tr_mask].min(), roll_vol[tr_mask].max()

    norm_level = np.clip(
        (roll_level - min_lvl) / (max_lvl - min_lvl + 1e-9), 0, 1
    )
    norm_vol = np.clip((roll_vol - min_vol) / (max_vol - min_vol + 1e-9), 0, 1)

    stress = 0.6 * norm_level + 0.4 * norm_vol
    stress = (
        pd.Series(stress).rolling(168, min_periods=24).mean().bfill().values
    )

    q_elev, q_crisis = np.quantile(stress[tr_mask], [0.55, 0.85])
    regime = np.where(stress > q_crisis, 2, np.where(stress > q_elev, 1, 0))

    df["stress"] = stress
    df["regime"] = regime

    names = {0: "Normal", 1: "Elevated", 2: "Crisis"}
    for r in [0, 1, 2]:
        m = regime == r
        yr = pd.Series(df["datetime"][m]).dt.year.mode()
        print(
            f"  {names[r]:9s}: {m.sum():6,} hrs  avg €{price[m].mean():6.1f}  "
            f"(mostly {yr.iloc[0] if len(yr) else '?'})"
        )

    crisis_2022 = ((regime == 2) & (df["datetime"].dt.year == 2022)).sum()
    crisis_total = (regime == 2).sum()
    print(
        f"  >> {crisis_2022:,} of {crisis_total:,} crisis hours are in 2022 "
        f"({crisis_2022/crisis_total*100:.0f}%) — matches the real energy crisis!"
    )
    return df, q_elev, q_crisis


def visualize_regimes_plotly(df, q_elev, q_crisis, save_path=os.path.join(DOCS_DIR, "part7_regimes.html")):
    print("Visualizing regimes with Plotly...")
    df["datetime"] = pd.to_datetime(df["datetime"])
    dates = df["datetime"]
    price = df["price"].values
    stress = df["stress"].values
    regime = df["regime"].values

    rcol = {0: "#1D9E75", 1: "#BA7517", 2: "#D85A30"}
    names = {0: "Normal", 1: "Elevated", 2: "Crisis"}

    fig = make_subplots(
        rows=3,
        cols=1,
        subplot_titles=(
            "Price timeline — the model paints 2022 RED automatically (the crisis)",
            "The grid-stress signal — crosses the red line during the crisis",
            "Regime classification of every hour",
        ),
        row_heights=[0.55, 0.30, 0.15],
        vertical_spacing=0.08,
        shared_xaxes=True,
    )

    fig.add_trace(
        go.Scatter(
            x=dates, y=price, mode="lines",
            line=dict(color="#222222", width=0.8),
            name="Price (€/MWh)",
            hovertemplate="Date: %{x}<br>Price: €%{y:.2f}<extra></extra>",
        ),
        row=1, col=1,
    )

    shapes = []
    for r in [0, 1, 2]:
        inreg = (regime == r).astype(int)
        edges = np.where(np.diff(np.concatenate([[0], inreg, [0]])) != 0)[0]
        for i in range(0, len(edges), 2):
            s, e = edges[i], min(edges[i + 1] - 1, len(dates) - 1)
            shapes.append(
                dict(
                    type="rect", xref="x", yref="y",
                    x0=dates.iloc[s], x1=dates.iloc[e],
                    y0=price.min() - 50, y1=price.max() + 100,
                    fillcolor=rcol[r], opacity=0.15, layer="below", line_width=0,
                )
            )

    fig.add_annotation(
        x=pd.Timestamp("2022-08-01"), y=450,
        ax=pd.Timestamp("2023-06-01"), ay=600,
        xref="x", yref="y", axref="x", ayref="y",
        text="2022 crisis<br>auto-detected",
        showarrow=True, arrowhead=2, arrowsize=1, arrowwidth=1.5,
        arrowcolor="#791F1F", font=dict(size=12, color="#791F1F", family="sans-serif"),
        bgcolor="white", bordercolor="#791F1F", borderpad=4,
        row=1, col=1,
    )

    fig.add_trace(
        go.Scatter(
            x=dates, y=stress, mode="lines",
            line=dict(color="#185FA5", width=1.2),
            fill="tozeroy", fillcolor="rgba(55, 138, 221, 0.25)",
            name="Stress Index",
            hovertemplate="Date: %{x}<br>Stress: %{y:.3f}<extra></extra>",
        ),
        row=2, col=1,
    )

    fig.add_hline(y=q_elev, line_dash="dash", line_color="#BA7517", line_width=1.5, annotation_text="Elevated threshold", annotation_position="top right", row=2, col=1)
    fig.add_hline(y=q_crisis, line_dash="dash", line_color="#D85A30", line_width=1.5, annotation_text="Crisis threshold", annotation_position="top right", row=2, col=1)

    for r in [0, 1, 2]:
        inreg = (regime == r).astype(int)
        edges = np.where(np.diff(np.concatenate([[0], inreg, [0]])) != 0)[0]
        for i in range(0, len(edges), 2):
            s, e = edges[i], min(edges[i + 1] - 1, len(dates) - 1)
            shapes.append(
                dict(
                    type="rect", xref="x3", yref="y3",
                    x0=dates.iloc[s], x1=dates.iloc[e],
                    y0=0, y1=1, fillcolor=rcol[r], opacity=0.85, layer="below", line_width=0,
                )
            )

    for r in [0, 1, 2]:
        fig.add_trace(
            go.Scatter(
                x=[None], y=[None], mode="markers",
                marker=dict(size=12, color=rcol[r], symbol="square"),
                name=names[r], showlegend=True,
            ),
            row=1, col=1,
        )

    fig.update_yaxes(title_text="Price (€/MWh)", range=[price.min() - 50, price.max() + 100], row=1, col=1)
    fig.update_yaxes(title_text="Stress index", range=[0, 1.05], row=2, col=1)
    fig.update_yaxes(showticklabels=False, showgrid=False, range=[0, 1], row=3, col=1)
    fig.update_xaxes(title_text="Date", row=3, col=1)

    fig.update_layout(
        shapes=shapes,
        title=dict(
            text="Automatic Regime Detection in German Electricity Prices",
            font=dict(size=18), x=0.5, xanchor="center",
        ),
        paper_bgcolor="#f7f6f3", plot_bgcolor="#ffffff",
        height=1000, width=1300,
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="center", x=0.5, font=dict(size=12)),
    )
    fig.update_xaxes(title_text="Date", type="date", tickformat="%Y", dtick="M12", row=3, col=1)
    fig.update_xaxes(showgrid=True, gridwidth=1, gridcolor="rgba(0,0,0,0.1)")
    fig.update_yaxes(showgrid=True, gridwidth=1, gridcolor="rgba(0,0,0,0.1)", row=1, col=1)
    fig.update_yaxes(showgrid=True, gridwidth=1, gridcolor="rgba(0,0,0,0.1)", row=2, col=1)

    fig.write_html(save_path)
    fig.show()


if __name__ == "__main__":
    df = pd.read_pickle(os.path.join(DATA_DIR, "data_part2.pkl"))
    df, q_elev, q_crisis = detect_regimes(df)
    visualize_regimes_plotly(df, q_elev, q_crisis)
    df.to_pickle(os.path.join(DATA_DIR, "data_part7_regimes.pkl"))
    print("\n✓ Saved regimes — ready for Parts 9 & 11")