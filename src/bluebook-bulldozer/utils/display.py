import pathlib
import pandas as pd
import streamlit as st

from typing import Callable
from decorator import decorator
from functools import singledispatch
from pandas.io.formats.style import Styler

# import from typehint
from .typehint import P, R


DF_STYLE = {
    "border-width": "2.5px",
    "font-family": "system-ui",
    "font-size": "14.5px",
}


def display_header(header: str, level: int = 1) -> None:
    prefix = "#" * level
    st.markdown(f"{prefix} {header}")


def display_content(content_file: str | pathlib.Path) -> None:
    with open(content_file) as f:
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
    ...


@set_style.register
def _(styler: Styler) -> Styler:
    return styler.pipe(
        lambda s: s.format(precision=4, na_rep=" ").set_properties(**DF_STYLE)
    )


@set_style.register
def _(df: pd.DataFrame) -> Styler:
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
