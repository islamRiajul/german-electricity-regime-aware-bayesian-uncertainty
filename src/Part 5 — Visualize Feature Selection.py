import os
import numpy as np
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from setup_imports import DATA_DIR, DOCS_DIR

def visualize_feature_selection_plotly(data, save_path=os.path.join(DOCS_DIR, "part4_feature_selection.html")):
    print("Creating feature selection visualization with Plotly...")

    features = data["features"]
    votes = data["votes"]
    scores = data["scores"]
    selected = data["selected"]
    removed = data["removed"]

    order = sorted(features, key=lambda f: votes[f])
    n = len(order)

    def get_color(f):
        if f in removed:
            return "#D85A30"
        if f in selected:
            return "#1D9E75"
        return "#C0BDB4"

    colors = [get_color(f) for f in order]

    fig = make_subplots(
        rows=2,
        cols=2,
        subplot_titles=(
            "1. Pearson correlation with price",
            "2. Random Forest importance",
            "3. All 12 methods (brighter = more important)",
            "4. Final vote count — the decision",
        ),
        horizontal_spacing=0.18,
        vertical_spacing=0.12,
    )

    vals1 = [scores["Pearson"].get(f, 0) for f in order]
    fig.add_trace(
        go.Bar(
            x=vals1,
            y=order,
            orientation="h",
            marker_color=colors,
            name="Pearson",
            showlegend=False,
            hovertemplate="Feature: %{y}<br>Pearson: %{x:.4f}<extra></extra>",
        ),
        row=1, col=1,
    )
    fig.add_vline(x=0.15, line_dash="dash", line_color="red", line_width=1.5, annotation_text="threshold 0.15", annotation_position="bottom right", row=1, col=1)

    vals2 = [scores["RF"].get(f, 0) for f in order]
    fig.add_trace(
        go.Bar(
            x=vals2,
            y=order,
            orientation="h",
            marker_color=colors,
            name="RF",
            showlegend=False,
            hovertemplate="Feature: %{y}<br>Importance: %{x:.4f}<extra></extra>",
        ),
        row=1, col=2,
    )

    methods = [
        "Pearson", "Spearman", "MutualInfo", "F-stat", "RFE", "SFS",
        "LASSO", "ElasticNet", "RF", "Permutation", "Granger", "SHAP",
    ]
    M = np.zeros((n, len(methods)))
    for j, m in enumerate(methods):
        raw = np.array([scores[m].get(f, 0) for f in order])
        max_val = raw.max()
        M[:, j] = raw / (max_val + 1e-9)

    fig.add_trace(
        go.Heatmap(
            z=M,
            x=methods,
            y=order,
            colorscale="YlOrRd",
            zmin=0,
            zmax=1,
            showscale=True,
            colorbar=dict(title="Normalized", len=0.42, y=0.2, x=0.45, thickness=15),
            hovertemplate="Feature: %{y}<br>Method: %{x}<br>Score: %{z:.2f}<extra></extra>",
        ),
        row=2, col=1,
    )

    vals4 = [votes[f] for f in order]
    fig.add_trace(
        go.Bar(
            x=vals4,
            y=order,
            orientation="h",
            marker_color=colors,
            text=[str(v) for v in vals4],
            textposition="outside",
            name="Votes",
            showlegend=False,
            hovertemplate="Feature: %{y}<br>Votes: %{x}<extra></extra>",
        ),
        row=2, col=2,
    )
    fig.add_vline(x=5, line_dash="dash", line_color="red", line_width=1.5, annotation_text="threshold (>=5)", annotation_position="bottom right", row=2, col=2)

    fig.add_trace(go.Bar(x=[None], y=[None], marker_color="#1D9E75", name="Selected", showlegend=True), row=1, col=1)
    fig.add_trace(go.Bar(x=[None], y=[None], marker_color="#C0BDB4", name="Dropped (too few votes)", showlegend=True), row=1, col=1)
    fig.add_trace(go.Bar(x=[None], y=[None], marker_color="#D85A30", name="Removed by VIF (redundant)", showlegend=True), row=1, col=1)

    fig.update_xaxes(title_text="|correlation|", row=1, col=1)
    fig.update_xaxes(title_text="importance", row=1, col=2)
    fig.update_xaxes(title_text="number of methods voting yes", range=[0, len(methods) + 2], row=2, col=2)

    fig.update_layout(
        title=dict(
            text=f"<b>Feature Selection — {len(selected)} of {len(features)} features selected (13 methods, real SMARD data)</b>",
            font=dict(size=16),
            x=0.5,
            xanchor="center",
        ),
        paper_bgcolor="#f7f6f3",
        plot_bgcolor="#ffffff",
        height=1200,
        width=1400,
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="center", x=0.5, font=dict(size=11)),
    )

    fig.update_xaxes(showgrid=True, gridwidth=1, gridcolor="rgba(0,0,0,0.1)")
    fig.update_yaxes(showgrid=True, gridwidth=1, gridcolor="rgba(0,0,0,0.1)")

    fig.write_html(save_path)
    print(f"  ✓ Saved feature selection dashboard to {save_path}")
    fig.show()


if __name__ == "__main__":
    file_path = os.path.join(DATA_DIR, "fs_part4.npy")
    print(f"Loading feature selection data from {file_path}...")
    if os.path.exists(file_path):
        data = np.load(file_path, allow_pickle=True).item()
        visualize_feature_selection_plotly(data)
    else:
        print(f"Error: Could not find {file_path}. Make sure Part 4 has been run and saved.")