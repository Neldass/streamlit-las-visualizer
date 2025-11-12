from __future__ import annotations

from typing import Optional, List, Dict

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots


def plot_crossplot(
    df: pd.DataFrame,
    x: str,
    y: str,
    color: Optional[str] = None,
    use_density: bool = False,
    title: str = "Crossplot",
):
    clean = df[[x, y] + ([color] if color else [])].replace([np.inf, -np.inf], np.nan).dropna()

    if use_density and color is None:
        fig = px.density_heatmap(
            clean, x=x, y=y, nbinsx=60, nbinsy=60, color_continuous_scale="Viridis"
        )
        fig.update_traces(hovertemplate=f"{x}=%{{x}}<br>{y}=%{{y}}<br>count=%{{z}}<extra></extra>")
    else:
        fig = px.scatter(
            clean,
            x=x,
            y=y,
            color=color,
            render_mode="webgl",
            opacity=0.7,
        )
    fig.update_layout(title=title, height=700)
    return fig


def plot_logs_custom_tracks(
    df: pd.DataFrame,
    track_groups: List[List[str]],
    depth_col: str = "DEPTH",
    units: Optional[Dict[str, Optional[str]]] = None,
    title: str = "Tracks (custom)",
    track_width: int = 270,
    height: int = 750,
) -> go.Figure:
    """Render user-defined tracks: each item in track_groups is a list of logs to overlay in one track.

    - track_groups: e.g., [["RHOB"], ["DT", "NPHI"], ["GR"]]
    - Units are used only for legend labels.
    """
    groups = [list(g) for g in track_groups if g]
    n = len(groups)
    units = units or {}
    if n == 0:
        return go.Figure()

    fig = make_subplots(rows=1, cols=n, shared_yaxes=True, horizontal_spacing=0.04)
    depth = df[depth_col]
    color_cycle = px.colors.qualitative.D3
    for col_idx, logs in enumerate(groups, start=1):
        for i, log in enumerate(logs):
            if log not in df.columns:
                continue
            series = df[log]
            color = color_cycle[(col_idx * 5 + i) % len(color_cycle)]
            unit = units.get(log)
            label = f"{log}{f' ({unit})' if unit else ''}"
            fig.add_trace(
                go.Scatter(
                    x=series,
                    y=depth,
                    mode="lines",
                    name=label,
                    line=dict(width=1.2, color=color),
                    showlegend=True,
                ),
                row=1,
                col=col_idx,
            )
        # Axis title lists logs in track
        track_title = ", ".join([l for l in logs if l in df.columns])
        fig.update_xaxes(title_text=track_title, row=1, col=col_idx)

    # Vertical separators
    total_width = max(500, track_width * n)
    for border in range(1, n):
        x_pos = border / n
        fig.add_shape(
            type="line",
            xref="paper",
            yref="paper",
            x0=x_pos,
            x1=x_pos,
            y0=0,
            y1=1,
            line=dict(color="rgba(150,150,150,0.4)", width=2, dash="dot"),
            layer="below",
        )

    # Y axis only on first column
    for i in range(1, n + 1):
        fig.update_yaxes(
            autorange="reversed",
            title_text=depth_col if i == 1 else None,
            showticklabels=True if i == 1 else False,
            showgrid=True,
            gridcolor="rgba(180,180,180,0.35)",
            gridwidth=0.7,
            zeroline=False,
            row=1,
            col=i,
        )
        fig.update_xaxes(
            showgrid=True,
            gridcolor="rgba(200,200,200,0.3)",
            gridwidth=0.6,
            zeroline=False,
            row=1,
            col=i,
        )
    fig.update_layout(
        title=title,
        height=height,
        width=total_width,
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
        margin=dict(l=40, r=20, t=60, b=40),
    )
    return fig
