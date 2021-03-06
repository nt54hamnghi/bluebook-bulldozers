import random
from time import time

import numpy as np
import pandas as pd
import streamlit as st

import utils as ut
from stages.eda import categorical, numerical


def run(dataframe: pd.DataFrame) -> None:
    st.header("Explanatory Data Analysis")
    with st.container():

        sections = st.selectbox(
            "Select Sections",
            options=["Pre-Analysis", "Numerical", "Categorical"],
        )
        random_sample = st.checkbox("Resample Each Time")

        with st.expander("Note:"):
            st.write(
                """
                This analysis use a random sample from the original data set.
                It has 45,000 rows and stays static throughout multiple runs.
                To generate new sample set each run.
                Click the checkbox above.
            """
            )

        dataframe.replace(
            {"YearMade": 1000, "MachineHoursCurrentMeter": 0},
            value=np.nan,
            inplace=True,
        )
        random_state = random.randint(0, int(time())) if random_sample else 0
        dataframe = dataframe.sample(
            frac=0.125, replace=True, random_state=random_state
        )
        if sections == "Pre-Analysis":
            st.subheader(f"1. {sections}")
            ut.display_header("Removing Unnecessary Variables", 4)
            ut.display_content("dropcols.txt")

            ut.display_header("Investigating Feature Summary", 4)
            uniques = dataframe.select_dtypes(include=["number"]).nunique()
            uniques.name = "unique"
            ut.styleit(st.table)(dataframe.describe().append(uniques))
            ut.display_content("summary.txt")

        elif sections == "Numerical":
            numerical.run(dataframe)
        elif sections == "Categorical":
            categorical.run(dataframe)
