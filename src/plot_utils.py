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
        fig = px.density_heatmap(clean, x=x, y=y, nbinsx=60, nbinsy=60, color_continuous_scale="Viridis")
        fig.update_traces(hovertemplate=f"{x}=%{{x}}<br>{y}=%{{y}}<br>count=%{{z}}<extra></extra>")
    else:
        fig = px.scatter(clean, x=x, y=y, color=color, render_mode="webgl", opacity=0.7, color_discrete_map=color_discrete_map)
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
    show_legend: bool = True,
    legend_position: str = "top",
) -> go.Figure:
    groups = [list(g) for g in track_groups if g]
    n = len(groups)
    units = units or {}
    if n == 0:
        return go.Figure()

    fig = make_subplots(rows=1, cols=n, shared_yaxes=True, horizontal_spacing=0.04)
    depth = df[depth_col]
    color_cycle = px.colors.qualitative.D3
    trimmed_ranges: Dict[int, Tuple[float, float]] = {}

    seen_legend: set[str] = set()

    for col_idx, logs in enumerate(groups, start=1):
        norm_ranges: Dict[str, Tuple[float, float]] = {}
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
                    legendgroup=label,
                    showlegend=(label not in seen_legend),
                    line=dict(color=color, width=line_width),
                    opacity=opacity,
                ),
                row=1,
                col=col_idx,
            )
            seen_legend.add(label)

        track_title = ", ".join([l for l in logs if l in df.columns])
        if normalize and track_title:
            track_title += " (norm)"
        auto_x_range: Optional[Tuple[float, float]] = None
        if len(logs) == 1 and logs[0] in df.columns:
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
                    fig.update_yaxes(range=[dmax, dmin], row=1, col=col_idx)
                    trimmed_ranges[col_idx] = (dmax, dmin)

    total_width = max(track_width, 120) if n == 1 else max(track_width * n, 120 * n)
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

    for i in range(1, n + 1):
        if i in trimmed_ranges:
            rng = list(trimmed_ranges[i])
            fig.update_yaxes(
                range=rng,
                autorange=False,
                title_text=depth_col if i == 1 else None,
                showticklabels=True if i == 1 else False,
                showgrid=True,
                gridcolor="rgba(180,180,180,0.35)",
                gridwidth=0.7,
                zeroline=False,
                row=1,
                col=i,
            )
        else:
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
        fig.update_xaxes(showgrid=True, gridcolor="rgba(200,200,200,0.3)", gridwidth=0.6, zeroline=False, row=1, col=i)

    # Legend configuration
    legend_cfg = dict()
    if legend_position == "top":
        legend_cfg = dict(orientation="h", yanchor="bottom", y=1.02, xanchor="left", x=0)
    elif legend_position == "bottom":
        legend_cfg = dict(orientation="h", yanchor="top", y=-0.08, xanchor="left", x=0)
    elif legend_position == "right":
        legend_cfg = dict(orientation="v", x=1.02, xanchor="left", y=1, yanchor="top")
    elif legend_position == "left":
        legend_cfg = dict(orientation="v", x=-0.02, xanchor="right", y=1, yanchor="top")
    else:
        legend_cfg = dict(orientation="h", yanchor="bottom", y=1.02, xanchor="left", x=0)

    fig.update_layout(
        title=title,
        height=height,
        width=total_width,
        showlegend=show_legend,
        legend=legend_cfg,
        margin=dict(l=40, r=20, t=70, b=60 if legend_position == "bottom" else 40),
    )
    return fig


def plot_logs_multi_wells(
    df_by_well: Dict[str, pd.DataFrame],
    track_groups: List[List[str]],
    depth_col: str = "DEPTH",
    units: Optional[Dict[str, Optional[str]]] = None,
    title: str = "Tracks (multi-well)",
    track_width: int = 270,
    height: int = 750,
    color_map: Optional[Dict[str, str]] = None,
    line_width: float = 1.2,
    normalize: bool = False,
    opacity: float = 1.0,
    x_ranges: Optional[List[Optional[Tuple[float, float]]]] = None,
    trim_depth_gaps: bool = False,
    show_legend: bool = True,
) -> go.Figure:
    """Render multiple wells side-by-side in a single row.

    Columns = n_wells * n_tracks. `track_width` is applied per track per well.
    """
    wells = list(df_by_well.keys())
    groups = [list(g) for g in track_groups if g]
    n_tracks = len(groups)
    n_wells = len(wells)
    if n_tracks == 0 or n_wells == 0:
        return go.Figure()

    total_cols = n_tracks * n_wells
    fig = make_subplots(rows=1, cols=total_cols, shared_yaxes=True, horizontal_spacing=0.04)
    units = units or {}
    color_cycle = px.colors.qualitative.D3

    # For separators between wells
    well_col_offsets: Dict[int, int] = {wi: wi * n_tracks for wi in range(n_wells)}

    # Show a single legend entry per well (even if multiple logs)
    seen_wells: set[str] = set()
    for wi, w in enumerate(wells):
        df = df_by_well[w]
        depth = df[depth_col]
        trimmed_ranges: Dict[int, Tuple[float, float]] = {}

        for ti, logs in enumerate(groups):
            col_idx = well_col_offsets[wi] + ti + 1

            # Optional normalization per well
            norm_ranges: Dict[str, Tuple[float, float]] = {}
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
                if normalize and log in norm_ranges:
                    vmin, vmax = norm_ranges[log]
                    series = (pd.to_numeric(series, errors="coerce") - vmin) / (vmax - vmin)
                color = (color_map.get(log) if (color_map and log in color_map) else color_cycle[(ti * 5 + i) % len(color_cycle)])
                unit = units.get(log)
                # Legend label groups by well only to avoid duplicates
                well_label = str(w)
                trace_name = well_label
                if normalize and log in norm_ranges:
                    trace_name += " [norm]"
                fig.add_trace(
                    go.Scatter(
                        x=series,
                        y=depth,
                        mode="lines",
                        name=trace_name,
                        legendgroup=well_label,
                        showlegend=(well_label not in seen_wells),
                        line=dict(color=color, width=line_width),
                        opacity=opacity,
                    ),
                    row=1,
                    col=col_idx,
                )
                seen_wells.add(well_label)

            track_title = ", ".join([l for l in logs if l in df.columns])
            if normalize and track_title:
                track_title += " (norm)"

            auto_x_range: Optional[Tuple[float, float]] = None
            if len(logs) == 1 and logs[0] in df.columns:
                s_vals = pd.to_numeric(df[logs[0]], errors="coerce").replace([np.inf, -np.inf], np.nan).dropna()
                if not s_vals.empty:
                    vmin = float(s_vals.min())
                    vmax = float(s_vals.max())
                    if vmax > vmin:
                        auto_x_range = (vmin, vmax)

            fig.update_xaxes(title_text=(track_title if wi == 0 else None), row=1, col=col_idx)
            applied_manual = False
            if x_ranges and len(x_ranges) >= (ti + 1):
                xr = x_ranges[ti]
                if xr and isinstance(xr, tuple) and len(xr) == 2:
                    x0, x1 = xr
                    if x0 is not None and x1 is not None and x1 > x0:
                        fig.update_xaxes(range=[float(x0), float(x1)], row=1, col=col_idx)
                        applied_manual = True
            if (not applied_manual) and auto_x_range:
                fig.update_xaxes(range=list(auto_x_range), row=1, col=col_idx)

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
                        fig.update_yaxes(range=[dmax, dmin], row=1, col=col_idx)
                        trimmed_ranges[col_idx] = (dmax, dmin)

    # Grid and y-axes settings
    for c in range(1, total_cols + 1):
        fig.update_yaxes(
            title_text=depth_col if c == 1 else None,
            showticklabels=True if c == 1 else False,
            showgrid=True,
            gridcolor="rgba(180,180,180,0.35)",
            gridwidth=0.7,
            zeroline=False,
            row=1,
            col=c,
            autorange="reversed",
        )
        fig.update_xaxes(
            showgrid=True,
            gridcolor="rgba(200,200,200,0.3)",
            gridwidth=0.6,
            zeroline=False,
            row=1,
            col=c,
        )

    # Visual separators between wells (thicker)
    total_width = max(track_width * n_tracks * n_wells, 120 * total_cols)
    if n_wells > 1:
        for wi in range(1, n_wells):
            x_pos = (wi * n_tracks) / total_cols
            fig.add_shape(
                type="line",
                xref="paper",
                yref="paper",
                x0=x_pos,
                x1=x_pos,
                y0=0,
                y1=1,
                line=dict(color="rgba(120,120,120,0.5)", width=3, dash="dot"),
                layer="below",
            )

    fig.update_layout(
        title=title,
        height=height,
        width=total_width,
        showlegend=show_legend,
        legend=dict(orientation="h", yanchor="top", y=-0.08, xanchor="left", x=0),
        margin=dict(l=40, r=20, t=70, b=80),
    )
    # Annotation retirée pour éviter double affichage du nom dans cas single-track.
    return fig
