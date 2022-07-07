import itertools
from typing import Any, Callable, NamedTuple, Optional

import colormap as cm
import numpy as np
import pandas as pd
import plotly.express as px
import seaborn as sns
import streamlit as st
from plotly.basedatatypes import BaseTraceType
from plotly.graph_objects import Figure
from plotly.subplots import make_subplots

PRIMARY_COLOR = "#636efa"
CM_BLUE = sns.dark_palette(color=PRIMARY_COLOR, as_cmap=True)
BASE_LAYOUT = dict(
    autosize=False, height=600, margin=dict(l=1, r=1, b=1, t=1)
)


class AxesLabel(NamedTuple):
    xlabel: str
    ylabel: str


class TraceContainer(NamedTuple):
    graph: BaseTraceType | list[BaseTraceType]
    labels: AxesLabel


def cmap2hexlist(
    color: str = PRIMARY_COLOR, tone: str = "dark", n_colors: int = 1500
) -> list[str]:

    assert tone in ["dark", "light"]

    palette = sns.dark_palette if tone == "dark" else sns.light_palette
    cmap = palette(color, as_cmap=True)
    hexlist = np.empty((n_colors + 1,), dtype="object")

    for i in range(n_colors + 1):
        rgb = (np.array(cmap(i)) * 255).astype(int)[:-1]
        hex = cm.rgb2hex(*rgb)
        hexlist[i] = hex
    return hexlist.tolist()


def render(fig: Figure, layout=None) -> None:
    layout = BASE_LAYOUT if layout is None else BASE_LAYOUT | layout
    fig.update_layout(**layout)
    st.plotly_chart(fig, use_container_width=True)


def _create_axes(nrows: int, ncols: int) -> np.ndarray:
    row = np.arange(1, nrows + 1)
    col = np.arange(1, ncols + 1)
    return np.array(list(itertools.product(row, col)))


def subplots(
    trace_container: list[TraceContainer], nrows: int, ncols: int
) -> Figure:

    # check if the number of subplots is equal to the number of traces
    if len(trace_container) != ncols * nrows:
        raise ValueError(
            "Number of graphs is not compatible to number of subplots."
        )
    fig = make_subplots(rows=nrows, cols=ncols)
    axes = _create_axes(nrows, ncols)
    for trace, ax in zip(trace_container, axes):
        # if it's a single trace
        if isinstance(trace.graph, BaseTraceType):
            fig.add_trace(trace.graph, row=ax[0], col=ax[1])
        else:
            for graph in trace.graph:
                fig.add_trace(graph, row=ax[0], col=ax[1])
        fig.update_xaxes(title_text=trace.labels.xlabel, row=ax[0], col=ax[1])
        fig.update_yaxes(title_text=trace.labels.ylabel, row=ax[0], col=ax[1])
    return fig


def identity_fn(x: Any) -> Any:
    return x


def overlay_histogram(
    df: pd.DataFrame,
    x: str,
    hue: str,
    transform_fn: Callable[..., Any] = identity_fn,
    to_replace: Optional[dict[str, Any]] = None,
    **kwds
) -> list[BaseTraceType]:

    tmp = df[[x, hue]].dropna()
    color = tmp[hue]
    if to_replace is not None:
        color = color.replace(to_replace)
    fig = px.histogram(x=transform_fn(tmp[x]), color=color, nbins=30, **kwds)
    return list(fig.select_traces())


def overlay_boxplot(
    df: pd.DataFrame,
    x: str,
    hue: str,
    transform_fn: Callable[..., Any] = identity_fn,
    to_replace: Optional[dict[str, Any]] = None,
    **kwds
) -> list[BaseTraceType]:

    tmp = df[[x, hue]].dropna()
    color = tmp[hue]
    if to_replace is not None:
        color = color.replace(to_replace)
    fig = px.box(x=transform_fn(tmp[x]), y=color, color=color, **kwds)
    return list(fig.select_traces())
