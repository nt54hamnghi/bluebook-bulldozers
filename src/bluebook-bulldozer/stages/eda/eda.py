import numpy as np
import utils as ut
import pandas as pd
import streamlit as st
import plotly.graph_objects as go

from dirs import CONTENT_DIR


def run(train: pd.DataFrame, valid: pd.DataFrame) -> None:
    st.header("Explanatory Data Analysis")

    inspect = st.container()
    numvars = st.container()
    # catvars = st.container()
    # datevars = st.container()

    with inspect:
        st.subheader("1. Initial Inspection")
        ut.display_header("Removing Unnecessary Variables", 4)
        ut.display_content(CONTENT_DIR / "dropcols.txt")

        ut.display_header("Investigating Feature Summary", 4)
        uniques = train.select_dtypes(include=["number"]).nunique()
        uniques.name = "unique"
        ut.styleit(st.table)(train.describe().append(uniques))
        ut.display_content(CONTENT_DIR / "summary.txt")

    train["YearMade"] = train["YearMade"].replace(1000, np.nan)

    with numvars:
        st.subheader("2. Numerical Variables")
        ut.display_header("`Yearmade`", 4)
        ut.display_content(CONTENT_DIR / "yearmade.txt")

        year_grps = train.groupby("YearMade")["SalePrice"].mean()

        boxplot = ut.Trace(
            go.Box(
                x=train["YearMade"], boxpoints="outliers",
                quartilemethod="inclusive",
                name=""
            ),
            ut.AxesLabel("Year Made", "")
        )
        lineplot = ut.Trace(
            go.Scatter(
                x=year_grps.index, y=year_grps,
                mode="lines", name="lines"
            ),
            ut.AxesLabel("Year Made", "Mean Sale Price")
        )
        fig = ut.subplots([boxplot, lineplot], nrows=1, ncols=2)
        ut.render(fig, layout=dict(height=400, showlegend=False))
