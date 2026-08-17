import os
import numpy as np
import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from setup_imports import DATA_DIR, DOCS_DIR

def signed_log(p):
    return np.sign(p) * np.log1p(np.abs(p))


def inverse_signed_log(z, price_clip=(-600.0, 1000.0)):
    z_min = (
        -np.log1p(np.abs(price_clip[0]))
        if price_clip[0] < 0
        else np.log1p(price_clip[0])
    )
    z_max = np.log1p(price_clip[1])

    z_safe = np.clip(z, z_min, z_max)
    p = np.sign(z_safe) * np.expm1(np.abs(z_safe))
    return np.clip(p, price_clip[0], price_clip[1])


def visualize_transform_plotly(price, save_path=os.path.join(DOCS_DIR, "part6_transform.html")):
    print("Visualizing signed-log transform with Plotly...")
    z = signed_log(price)

    fig = make_subplots(
        rows=2,
        cols=2,
        subplot_titles=(
            "1. RAW price — wide range, extreme spikes",
            "2. SIGNED-LOG price — compact, easier to model",
            "3. The transform curve — squashes big values, keeps small ones",
            "4. Round-trip check — transform → inverse = original",
        ),
        horizontal_spacing=0.1,
        vertical_spacing=0.12,
    )

    fig.add_trace(
        go.Histogram(
            x=price,
            nbinsx=100,
            marker_color="#D85A30",
            opacity=0.8,
            name="Raw Price",
            hovertemplate="Price (€/MWh): %{x}<br>Count: %{y}<extra></extra>",
        ),
        row=1, col=1,
    )
    fig.add_vline(x=0, line_dash="dot", line_color="black", line_width=1, row=1, col=1)
    fig.add_annotation(
        text=f"range: {price.max()-price.min():.0f}<br>std: {price.std():.0f}",
        xref="x domain", yref="y domain", x=0.95, y=0.9,
        showarrow=False, bgcolor="white", bordercolor="#ccc", borderwidth=1,
        row=1, col=1,
    )

    fig.add_trace(
        go.Histogram(
            x=z,
            nbinsx=100,
            marker_color="#1D9E75",
            opacity=0.8,
            name="Signed-Log",
            hovertemplate="Signed-log: %{x:.2f}<br>Count: %{y}<extra></extra>",
        ),
        row=1, col=2,
    )
    fig.add_vline(x=0, line_dash="dot", line_color="black", line_width=1, row=1, col=2)
    fig.add_annotation(
        text=f"range: {z.max()-z.min():.1f}<br>std: {z.std():.2f}",
        xref="x domain", yref="y domain", x=0.95, y=0.9,
        showarrow=False, bgcolor="white", bordercolor="#ccc", borderwidth=1,
        row=1, col=2,
    )

    p_range = np.linspace(-500, 936, 500)
    z_range = signed_log(p_range)
    fig.add_trace(
        go.Scatter(
            x=p_range, y=z_range, mode="lines",
            line=dict(color="#0C447C", width=2.5),
            name="Transform Curve",
            hovertemplate="Price: €%{x:.1f}<br>Signed-log: %{y:.3f}<extra></extra>",
        ),
        row=2, col=1,
    )

    pts = [-500, -50, 0, 50, 500, 936]
    z_pts = [signed_log(p) for p in pts]
    fig.add_trace(
        go.Scatter(
            x=pts, y=z_pts, mode="markers+text",
            marker=dict(color="#D85A30", size=8),
            text=[f"€{p}" for p in pts],
            textposition="top left",
            name="Sample Points",
            showlegend=False,
        ),
        row=2, col=1,
    )

    recovered = inverse_signed_log(z)
    sample_size = min(3000, len(price))
    idx = np.random.choice(len(price), sample_size, replace=False)

    fig.add_trace(
        go.Scatter(
            x=price[idx], y=recovered[idx], mode="markers",
            marker=dict(color="#7F77DD", size=4, opacity=0.3),
            name="Recovered Points",
            hovertemplate="Original: €%{x:.2f}<br>Recovered: €%{y:.2f}<extra></extra>",
        ),
        row=2, col=2,
    )

    fig.add_trace(
        go.Scatter(
            x=[price.min(), price.max()], y=[price.min(), price.max()],
            mode="lines", line=dict(color="red", dash="dash", width=1.5),
            name="Perfect Recovery",
        ),
        row=2, col=2,
    )

    err = np.abs(price - recovered).max()
    fig.add_annotation(
        text=f"max error: {err:.2e}<br>(perfectly invertible)",
        xref="x domain", yref="y domain", x=0.05, y=0.9,
        showarrow=False, bgcolor="white", bordercolor="#ccc", borderwidth=1,
        row=2, col=2,
    )

    fig.update_xaxes(title_text="Price (€/MWh)", row=1, col=1)
    fig.update_yaxes(title_text="Count", row=1, col=1)
    fig.update_xaxes(title_text="signed-log price", row=1, col=2)
    fig.update_yaxes(title_text="Count", row=1, col=2)
    fig.update_xaxes(title_text="Price (€/MWh)", row=2, col=1)
    fig.update_yaxes(title_text="signed-log value", row=2, col=1)
    fig.update_xaxes(title_text="Original price", row=2, col=2)
    fig.update_yaxes(title_text="Recovered price", row=2, col=2)

    fig.update_layout(
        title=dict(
            text="Signed-Log Transform — Taming Negative Prices & Extreme Spikes",
            font=dict(size=16), x=0.5, xanchor="center",
        ),
        paper_bgcolor="#f7f6f3", plot_bgcolor="#ffffff",
        height=900, width=1200, showlegend=False,
    )

    fig.update_xaxes(showgrid=True, gridwidth=1, gridcolor="rgba(0,0,0,0.1)")
    fig.update_yaxes(showgrid=True, gridwidth=1, gridcolor="rgba(0,0,0,0.1)")

    fig.write_html(save_path)
    print(f"  ✓ Saved interactive HTML dashboard to {save_path}")
    fig.show()


if __name__ == "__main__":
    df = pd.read_pickle(os.path.join(DATA_DIR, "data_part2.pkl"))
    price = df["price"].values

    print("\nExample transformations:")
    print(f"  {'price':>8} {'signed_log':>12} {'recovered':>10}")
    for p in [-500, -50, -1, 0, 1, 50, 500, 936]:
        z = signed_log(p)
        print(f"  {p:>8.1f} {z:>12.3f} {inverse_signed_log(z):>10.1f}")

    visualize_transform_plotly(price)