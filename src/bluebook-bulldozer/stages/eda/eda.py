import utils
import pandas as pd
import streamlit as st

from dirs import CONTENT_DIR


def run(train: pd.DataFrame, valid: pd.DataFrame) -> None:
    st.header("Explanatory Data Analysis")

    inspect = st.container()
    # numvars = st.container()
    # catvars = st.container()
    # datevars = st.container()

    with inspect:
        st.subheader("1. Initial Inspection")
        utils.display_header("Removing Unnecessary Variables", 4)
        utils.display_content(CONTENT_DIR / "dropcols.txt")

        utils.display_header("Investigating Feature Summary", 4)
        uniques = train.select_dtypes(include=["number"]).nunique()
        uniques.name = "unique"
        utils.styleit(st.table)(train.describe().append(uniques))
        utils.display_content(CONTENT_DIR / "summary.txt")
