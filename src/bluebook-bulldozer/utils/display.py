import pathlib
from functools import singledispatch
from typing import Callable

import pandas as pd
import streamlit as st
from decorator import decorator
from pandas.io.formats.style import Styler

from dirs import CONTENT_DIR

from .typehint import P, R

DF_STYLE = {
    "border-width": "2.5px",
    "font-family": "system-ui",
    "font-size": "14.5px",
}


def display_header(header: str, level: int = 1) -> None:
    prefix = "#" * level
    st.markdown(f"{prefix} {header}")


def display_content(
    filename: str | pathlib.Path, default_dir: pathlib.Path = CONTENT_DIR
) -> None:
    if default_dir is not None:
        filename = default_dir / filename
    with open(filename) as f:
        st.markdown(f.read())


def display_metric(
    metric: float,
    label: str,
    name,
    smaller_better: bool = False,
    help: str = "",
) -> None:
    if smaller_better:
        caption, delta_color = ("Smaller is better.", "inverse")
    else:
        caption, delta_color = ("Larger is better.", "normal")

    st.metric(
        label,
        metric,
        delta=round(metric - st.session_state[name], 5),
        delta_color=delta_color,
    )

    caption += "\n" + help
    st.caption(caption)

    st.session_state[name] = metric


@singledispatch
def set_style(arg: pd.DataFrame | Styler) -> Styler:
    raise NotImplementedError(type(arg))


@set_style.register
def style_styler(styler: Styler) -> Styler:
    return styler.pipe(
        lambda s: s.format(precision=4, na_rep=" ").set_properties(**DF_STYLE)
    )


@set_style.register
def style_dataframe(df: pd.DataFrame) -> Styler:
    return df.style.format(precision=4, na_rep=" ").set_properties(**DF_STYLE)


@decorator
def styleit(fn: Callable[P, R], *args, **kwds) -> R:
    try:
        data = kwds.pop("data")
    except KeyError:
        data, *args = args
    styled = set_style(data)
    result = fn(styled, *args, **kwds)
    return result
