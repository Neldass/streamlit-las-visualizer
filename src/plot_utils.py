from __future__ import annotations

from typing import Optional, List, Dict, Tuple

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
    color_discrete_map: Optional[Dict[str, str]] = None,
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
            color_discrete_map=color_discrete_map,
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
    color_map: Optional[Dict[str, str]] = None,
    line_width: float = 1.2,
    normalize: bool = False,
    opacity: float = 1.0,
    x_ranges: Optional[List[Optional[Tuple[float, float]]]] = None,
    trim_depth_gaps: bool = False,
) -> go.Figure:
    """Render user-defined tracks.

    Parameters
    ----------
    track_groups : list of list
        Each inner list contains log names to overlay in one track.
    color_map : dict optional
        Mapping log -> CSS color. If absent a qualitative cycle is used.
    line_width : float
        Line width for all logs.
    normalize : bool
        If True min-max normalizes each selected log independently to [0,1] for better overlay comparison.
    opacity : float
        Global opacity applied to all line traces (0-1).
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
        # Track if we applied a trimmed depth range (so we don't later override it)
        # We'll store ranges in a dict outside loop; initialize if first iteration
        if 'trimmed_ranges' not in locals():
            trimmed_ranges: Dict[int, Tuple[float, float]] = {}
        # Optional normalization: compute per-log min/max inside current df subset
        norm_ranges: Dict[str, tuple[float, float]] = {}
        if normalize:
            for log in logs:
                if log in df.columns:
                    s = pd.to_numeric(df[log], errors="coerce")
                    vmin = s.min(skipna=True)
                    vmax = s.max(skipna=True)
                    if pd.notna(vmin) and pd.notna(vmax) and vmax != vmin:
                        norm_ranges[log] = (float(vmin), float(vmax))
        for i, log in enumerate(logs):
            if log not in df.columns:
                continue
            series = df[log]
            # Normalize if requested
            if normalize and log in norm_ranges:
                vmin, vmax = norm_ranges[log]
                series = (pd.to_numeric(series, errors="coerce") - vmin) / (vmax - vmin)
            color = (color_map.get(log) if (color_map and log in color_map) else color_cycle[(col_idx * 5 + i) % len(color_cycle)])
            unit = units.get(log)
            label = f"{log}{f' ({unit})' if unit else ''}"
            if normalize and log in norm_ranges:
                label += " [norm]"
            fig.add_trace(
                go.Scatter(
                    x=series,
                    y=depth,
                    mode="lines",
                    name=label,
                    showlegend=True,
                ),
                row=1,
                col=col_idx,
            )
        # Axis title lists logs in track
        track_title = ", ".join([l for l in logs if l in df.columns])
        if normalize and track_title:
            track_title += " (norm)"
        # Auto-range logic: if only one log in this track and no manual x-range, tighten X to its min/max
        auto_x_range: Optional[Tuple[float, float]] = None
        if len(logs) == 1 and logs[0] in df.columns:
            # Use non-null finite values
            s_vals = pd.to_numeric(df[logs[0]], errors="coerce").replace([np.inf, -np.inf], np.nan).dropna()
            if not s_vals.empty:
                vmin = float(s_vals.min())
                vmax = float(s_vals.max())
                if vmax > vmin:
                    auto_x_range = (vmin, vmax)

        fig.update_xaxes(title_text=track_title, row=1, col=col_idx)
        applied_manual = False
        if x_ranges and len(x_ranges) >= col_idx:
            xr = x_ranges[col_idx - 1]
            if xr and isinstance(xr, tuple) and len(xr) == 2:
                x0, x1 = xr
                if x0 is not None and x1 is not None and x1 > x0:
                    fig.update_xaxes(range=[float(x0), float(x1)], row=1, col=col_idx)
                    applied_manual = True
        if (not applied_manual) and auto_x_range:
            fig.update_xaxes(range=list(auto_x_range), row=1, col=col_idx)

        # Depth trimming: restrict Y axis to depths where at least one log has finite data
        if trim_depth_gaps:
            finite_mask = None
            for log in logs:
                if log in df.columns:
                    s_depth = pd.to_numeric(df[log], errors="coerce").replace([np.inf, -np.inf], np.nan)
                    mask = s_depth.notna()
                    finite_mask = mask if finite_mask is None else (finite_mask | mask)
            if finite_mask is not None and finite_mask.any():
                depth_vals = pd.to_numeric(depth, errors="coerce").replace([np.inf, -np.inf], np.nan)
                used_depths = depth_vals[finite_mask]
                dmin = float(used_depths.min())
                dmax = float(used_depths.max())
                if dmax > dmin:
                    # Reversed axis: range expects [max, min]
                    fig.update_yaxes(range=[dmax, dmin], row=1, col=col_idx)
                    trimmed_ranges[col_idx] = (dmax, dmin)

    # Vertical separators
    # Compute total width. For a single track we respect the requested track_width
    # instead of stretching to container width. For multiple tracks we multiply.
    if n == 1:
        total_width = max(track_width, 120)
    else:
        total_width = max(track_width * n, 120 * n)
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
        yaxis_kwargs = dict(
            title_text=depth_col if i == 1 else None,
            showticklabels=True if i == 1 else False,
            showgrid=True,
            gridcolor="rgba(180,180,180,0.35)",
            gridwidth=0.7,
            zeroline=False,
            row=1,
            col=i,
        )
        if 'trimmed_ranges' in locals() and i in trimmed_ranges:
            # Preserve manual reversed range
            yaxis_kwargs['range'] = list(trimmed_ranges[i])
            yaxis_kwargs['autorange'] = False
        else:
            yaxis_kwargs['autorange'] = "reversed"
        fig.update_yaxes(**yaxis_kwargs)
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
