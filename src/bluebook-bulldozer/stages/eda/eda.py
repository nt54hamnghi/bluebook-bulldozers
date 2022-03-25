import numpy as np
import utils as ut
import pandas as pd
import streamlit as st

from dirs import CONTENT_DIR
from stages.eda import numerical, categorical


def run(dataframe: pd.DataFrame) -> None:
    st.header("Explanatory Data Analysis")
    with st.container():

        sections = st.selectbox(
            "Select Sections",
            options=["Pre-Analysis", "Numerical", "Categorical"],
        )

        if sections == "Pre-Analysis":
            st.subheader(f"1. {sections}")
            ut.display_header("Removing Unnecessary Variables", 4)
            ut.display_content(CONTENT_DIR / "dropcols.txt")

            ut.display_header("Investigating Feature Summary", 4)
            uniques = dataframe.select_dtypes(include=["number"]).nunique()
            uniques.name = "unique"
            ut.styleit(st.table)(dataframe.describe().append(uniques))
            ut.display_content(CONTENT_DIR / "summary.txt")

            dataframe["YearMade"] = dataframe["YearMade"].replace(1000, np.nan)
            dataframe["MachineHoursCurrentMeter"] = dataframe[
                "MachineHoursCurrentMeter"
            ].replace(0, np.nan)
        elif sections == "Numerical":
            numerical.run(dataframe)
        elif sections == "categorical":
            categorical.run(dataframe)
