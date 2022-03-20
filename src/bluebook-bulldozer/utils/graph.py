import numpy as np
import colormap as cm
import seaborn as sns
import streamlit as st

# import plotly.graph_objects as go

from itertools import product
from typing import NamedTuple
from plotly.basedatatypes import BaseTraceType
from plotly.graph_objects import Figure
from plotly.subplots import make_subplots

PRIMARY_COLOR = "#636efa"
CM_BLUE = sns.dark_palette(color=PRIMARY_COLOR, as_cmap=True)
BASE_LAYOUT = dict(
    autosize=False, height=600, margin=dict(l=1, r=1, b=1, t=1)
)


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
    return np.array(list(product(row, col)))


class AxesLabel(NamedTuple):
    xlabel: str
    ylabel: str


class Trace(NamedTuple):
    graph: BaseTraceType
    labels: AxesLabel


def subplots(traces: list[Trace], nrows: int, ncols: int) -> Figure:

    if len(traces) != ncols * nrows:
        raise ValueError(
            "Number of graphs is not compatible to number of subplots."
        )
    fig = make_subplots(rows=nrows, cols=ncols, subplot_titles=["a", "b"])
    axes = _create_axes(nrows, ncols)
    for trace, ax in zip(traces, axes):
        fig.add_trace(trace.graph, row=ax[0], col=ax[1])
        fig.update_xaxes(
            title_text=trace.labels.xlabel,
            row=ax[0], col=ax[1]
        )
        fig.update_yaxes(
            title_text=trace.labels.ylabel,
            row=ax[0], col=ax[1]
        )
    return fig
