from typing import Any, Optional

import numpy as np
import pandas as pd
import streamlit as st

import utils as ut
from utils.graph import AxesLabel, TraceContainer


def run(dataframe: pd.DataFrame) -> None:
    st.subheader("3. Categorical Variables")

    # Enclosure
    ut.display_header("`Enclosure`", 4)
    to_replace = {"EROPS AC": "EROPS w AC", "None or Unspecified": "NO ROPS"}
    ut.display_content("enclosure.txt")
    categorical_graphs(dataframe, "Enclosure", to_replace)

    # UsageBand
    ut.display_header("`UsageBand`", 4)
    ut.display_content("usageband.txt")
    categorical_graphs(dataframe, "UsageBand")

    # Drive_System
    ut.display_header("`Drive_System`", 4)
    ut.display_content("drivesys.txt")
    categorical_graphs(dataframe, "Drive_System")


def categorical_graphs(
    df: pd.DataFrame, cat: str, to_replace: Optional[dict[str, Any]] = None
) -> None:
    hist = TraceContainer(
        ut.overlay_histogram(
            df,
            x="SalePrice",
            hue=cat,
            to_replace=to_replace,
            transform_fn=np.log1p,
        ),
        AxesLabel("Log of Sale Price", "Count"),
    )

    box = TraceContainer(
        ut.overlay_boxplot(
            df,
            x="SalePrice",
            hue=cat,
            to_replace=to_replace,
            transform_fn=np.log1p,
        ),
        AxesLabel("Log of Sale Price", ""),
    )

    fig = ut.subplots([box, hist], nrows=1, ncols=2)

    ut.render(fig, layout=dict(height=400))
