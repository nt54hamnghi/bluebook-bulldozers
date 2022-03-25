import numpy as np
import pandas as pd
import utils as ut
import streamlit as st

from typing import Any, Optional
from utils.graph import TraceContainer, AxesLabel


def run(train: pd.Series) -> None:
    st.subheader("3. Categorical Variables")

    # ProductSize
    ut.display_header("`ProductSize`", 4)
    categorical_graphs(train, "ProductSize")

    # Enclosure
    ut.display_header("`Enclosure`", 4)
    to_replace = {"EROPS AC": "EROPS w AC", "None or Unspecified": "NO ROPS"}
    categorical_graphs(train, "Enclosure", to_replace)


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
        AxesLabel("Log of Sale Price", "Count"),
    )

    fig = ut.subplots([hist, box], nrows=1, ncols=2)

    ut.render(fig, layout=dict(height=550))