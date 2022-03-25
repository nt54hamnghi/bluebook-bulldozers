import random
import numpy as np
import utils as ut
import pandas as pd
import streamlit as st

from time import time
from dirs import CONTENT_DIR
from stages.eda import numerical, categorical


def run(dataframe: pd.DataFrame) -> None:
    st.header("Explanatory Data Analysis")
    with st.container():

        sections = st.selectbox(
            "Select Sections",
            options=["Pre-Analysis", "Numerical", "Categorical"],
        )
        random_sample = st.checkbox("Resample Each Time")
        random_state = random.randint(0, int(time())) if random_sample else 0
        dataframe = dataframe.sample(
            frac=0.25, replace=False, random_state=random_state
        )

        with st.expander("Note:"):
            st.write("""
                This analysis use a random sample from the original data set.
                It has 100,000 rows and stays static throughout multiple runs.
                To generate new sample set each run.
                Click the checkbox above.
            """)

        if sections == "Pre-Analysis":
            st.subheader(f"1. {sections}")
            ut.display_header("Removing Unnecessary Variables", 4)
            ut.display_content(CONTENT_DIR / "dropcols.txt")

            ut.display_header("Investigating Feature Summary", 4)
            uniques = dataframe.select_dtypes(include=["number"]).nunique()
            uniques.name = "unique"
            ut.styleit(st.table)(dataframe.describe().append(uniques))
            ut.display_content(CONTENT_DIR / "summary.txt")

            dataframe["YearMade"] = dataframe["YearMade"].replace(
                1000, np.nan
            )
            dataframe["MachineHoursCurrentMeter"] = dataframe[
                "MachineHoursCurrentMeter"
            ].replace(0, np.nan)
        elif sections == "Numerical":
            numerical.run(dataframe)
        elif sections == "Categorical":
            categorical.run(dataframe)
