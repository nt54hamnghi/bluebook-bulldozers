import numpy as np
import pandas as pd
import utils as ut
import streamlit as st
import plotly.graph_objects as go

from dirs import CONTENT_DIR
from utils.graph import TraceContainer, AxesLabel
from sklearn.preprocessing import minmax_scale


def run(dataframe: pd.Series) -> None:
    y = dataframe["SalePrice"]
    st.subheader("2. Numerical Variables")

    # YearMade
    ut.display_header("`YearMade`", 4)
    ut.display_content(CONTENT_DIR / "yearmade.txt")
    year_grps = dataframe.groupby("YearMade")["SalePrice"].mean()
    boxplot = TraceContainer(
        go.Box(
            x=dataframe["YearMade"],
            boxpoints="outliers",
            quartilemethod="inclusive",
        ),
        AxesLabel("Year Made", ""),
    )
    lineplot = TraceContainer(
        go.Scatter(x=year_grps.index, y=year_grps, mode="lines"),
        AxesLabel("Year Made", "Mean Sale Price"),
    )
    fig = ut.subplots([boxplot, lineplot], nrows=1, ncols=2)
    ut.render(fig, layout=dict(height=400, showlegend=False))

    # MachineHoursCurrentMeter
    ut.display_header("`MachineHoursCurrentMeter`", 4)
    ut.display_content(CONTENT_DIR / "machinehour-p1.txt")
    with st.expander("Note"):
        ut.display_content(CONTENT_DIR / "machinehour-p2.txt")
    x = dataframe["MachineHoursCurrentMeter"]
    pre_violon = TraceContainer(
        go.Violin(x=x, name="Pre-transformed"), AxesLabel("Hour", "")
    )
    pos_violin = TraceContainer(
        go.Violin(
            x=np.log1p(x),
            name="Post-transformed",
        ),
        AxesLabel("Log of Hour", ""),
    )
    fig = ut.subplots([pre_violon, pos_violin], nrows=1, ncols=2)
    ut.render(fig, layout=dict(height=400))
    scatter = go.Scattergl(
        x=np.log1p(x),
        y=y,
        mode="markers",
        marker=dict(line_width=0, size=2),
    )
    ut.display_content(CONTENT_DIR / "machinehour-p3.txt")
    ut.render(
        go.Figure(scatter),
        layout=dict(
            xaxis_title="Log of Hour",
            yaxis_title="Sale Price",
        ),
    )
    # MachineID
    ut.display_header("`MachineID`", 4)
    x = dataframe["MachineID"]
    pre_scatter = TraceContainer(
        go.Scattergl(
            x=x, y=y, mode="markers", marker=dict(line_width=0, size=2)
        ),
        AxesLabel("Machine ID", "Sale Price"),
    )

    pos_scatter = TraceContainer(
        go.Scattergl(
            x=minmax_scale(x.map(x.value_counts())),
            y=y,
            mode="markers",
            marker=dict(line_width=0, size=2),
        ),
        AxesLabel("Normalized Count by Machine ID", "Sale Price"),
    )
    fig = ut.subplots([pre_scatter, pos_scatter], nrows=1, ncols=2)
    ut.render(fig, layout=dict(height=400))
