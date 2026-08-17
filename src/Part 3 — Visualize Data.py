import os
import numpy as np
import pandas as pd
import webbrowser
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from setup_imports import DATA_DIR, DOCS_DIR

def visualize_data_plotly(df, save_path=os.path.join(DOCS_DIR, "part3_data_overview.html")):
    print("Creating data visualization with Plotly...")

    fig = make_subplots(
        rows=3,
        cols=2,
        subplot_titles=(
            "1. Price over time — the 2022 energy crisis is obvious",
            "2. Price distribution — note negative prices & long right tail",
            "3. Average price by hour — cheap at night, peaks morning & evening",
            "4. Average price by month — higher in winter",
            "5. Price vs renewable output — more renewables, lower price",
            "6. One example week (Jun 2023) — daily up-and-down rhythm",
        ),
        vertical_spacing=0.08,
        horizontal_spacing=0.08,
    )

    fig.add_trace(
        go.Scatter(
            x=df["datetime"],
            y=df["price"],
            mode="lines",
            line=dict(width=0.8, color="#378ADD"),
            name="Price (€/MWh)",
        ),
        row=1, col=1,
    )
    fig.add_hline(y=0, line_dash="dot", line_color="red", line_width=1, row=1, col=1)

    fig.add_trace(
        go.Histogram(
            x=df["price"],
            nbinsx=120,
            marker_color="#1D9E75",
            opacity=0.8,
            name="Distribution",
        ),
        row=1, col=2,
    )
    fig.add_vline(x=0, line_dash="dash", line_color="red", line_width=1.5, annotation_text="zero", row=1, col=2)
    fig.update_yaxes(type="log", row=1, col=2)

    hourly = df.groupby("hour")["price"].mean().reset_index()
    fig.add_trace(
        go.Scatter(
            x=hourly["hour"],
            y=hourly["price"],
            mode="lines+markers",
            line=dict(width=2, color="#BA7517"),
            fill="tozeroy",
            fillcolor="rgba(186, 117, 23, 0.2)",
            name="Avg Price",
        ),
        row=2, col=1,
    )

    monthly = df.groupby("month")["price"].mean().reset_index()
    months = ["Jan", "Feb", "Mar", "Apr", "May", "Jun", "Jul", "Aug", "Sep", "Oct", "Nov", "Dec"]
    fig.add_trace(
        go.Bar(
            x=months,
            y=monthly["price"],
            marker_color="#7F77DD",
            opacity=0.8,
            name="Avg Price",
        ),
        row=2, col=2,
    )

    renew = df["wind_offshore"] + df["wind_onshore"] + df["solar"]
    sample_size = min(4000, len(df))
    idx = np.random.choice(len(df), sample_size, replace=False)
    fig.add_trace(
        go.Scatter(
            x=renew.iloc[idx] / 1000,
            y=df["price"].iloc[idx],
            mode="markers",
            marker=dict(size=4, opacity=0.3, color="#D85A30"),
            name="Renewables vs Price",
        ),
        row=3, col=1,
    )
    fig.add_hline(y=0, line_dash="dot", line_color="red", line_width=1, row=3, col=1)

    week = df[(df["datetime"] >= "2023-06-05") & (df["datetime"] < "2023-06-12")]
    fig.add_trace(
        go.Scatter(
            x=week["datetime"],
            y=week["price"],
            mode="lines+markers",
            line=dict(width=1.2, color="#0C447C"),
            fill="tozeroy",
            fillcolor="rgba(12, 68, 124, 0.15)",
            name="Example Week",
        ),
        row=3, col=2,
    )

    fig.update_xaxes(title_text="Date", row=1, col=1)
    fig.update_yaxes(title_text="Price (€/MWh)", row=1, col=1)
    fig.update_xaxes(title_text="Price (€/MWh)", row=1, col=2)
    fig.update_yaxes(title_text="Number of hours (Log)", row=1, col=2)
    fig.update_xaxes(title_text="Hour of day", dtick=2, row=2, col=1)
    fig.update_yaxes(title_text="Avg price (€/MWh)", row=2, col=1)
    fig.update_xaxes(title_text="Month", row=2, col=2)
    fig.update_yaxes(title_text="Avg price (€/MWh)", row=2, col=2)
    fig.update_xaxes(title_text="Renewable generation (GW)", row=3, col=1)
    fig.update_yaxes(title_text="Price (€/MWh)", row=3, col=1)
    fig.update_xaxes(title_text="Date", tickformat="%a\n%b %d", row=3, col=2)
    fig.update_yaxes(title_text="Price (€/MWh)", row=3, col=2)

    fig.update_layout(
        title_text="German Day-Ahead Electricity Prices — Data Overview (2020–2026)",
        title_font=dict(size=18),
        paper_bgcolor="#f7f6f3",
        plot_bgcolor="#ffffff",
        height=1100,
        width=1300,
        showlegend=False,
    )
    fig.update_xaxes(showgrid=True, gridwidth=1, gridcolor="rgba(0,0,0,0.1)")
    fig.update_yaxes(showgrid=True, gridwidth=1, gridcolor="rgba(0,0,0,0.1)")

    fig.write_html(save_path)
    print(f"  ✓ Saved interactive HTML dashboard to {save_path}")
    try:
        fig.show(renderer="iframe")
    except Exception:
        webbrowser.open('file://' + os.path.abspath(save_path))


if __name__ == "__main__":
    df_path = os.path.join(DATA_DIR, "data_part2.pkl")
    print(f"Loading data from {df_path}...")
    df = pd.read_pickle(df_path)
    visualize_data_plotly(df)
    print("\n✓ Done — interactive dashboard saved and launched successfully!")