import numpy as np
import utils as ut
import pandas as pd
import streamlit as st

from dirs import CONTENT_DIR
from stages.eda import numerical


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
    train["MachineHoursCurrentMeter"] = train[
        "MachineHoursCurrentMeter"
    ].replace(0, np.nan)

    with numvars:
        numerical.run(train)
