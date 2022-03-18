import streamlit as st
import colormap as cm
import numpy as np
import seaborn as sns
import plotly.graph_objects as go


PRIMARY_COLOR = "#636efa"
CM_BLUE = sns.dark_palette(color=PRIMARY_COLOR, as_cmap=True)


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


def render(*graphs, **layout_kwargs) -> None:

    fig = go.Figure(data=graphs)
    fig.update_layout(
        **layout_kwargs, autosize=False, margin=dict(l=1, r=1, b=1, t=1)
    )
    st.plotly_chart(fig, use_container_width=True)
