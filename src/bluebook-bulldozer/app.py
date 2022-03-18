import pandas as pd
import streamlit as st
import stages

from dirs import DATA_DIR

__version__ = "0.0.1"
STAGES = ("Introduction", "EDA", "Modeling", "Feature Selection")


def _load_data(filename: str) -> pd.DataFrame:
    return pd.read_feather(DATA_DIR / filename)


@st.experimental_memo(show_spinner=False)
def _prepare_data() -> dict[str, tuple[pd.DataFrame, ...]]:
    # Raw data
    raw_train = _load_data("raw_train.feather")
    raw_valid = _load_data("raw_valid.feather")
    # Processed dataj
    xtrn = _load_data("xtrn.feather")
    xval = _load_data("xval.feather")
    ytrn = _load_data("ytrn.feather").iloc[:, 0]
    yval = _load_data("yval.feather").iloc[:, 0]

    return dict(raw=(raw_train, raw_valid), trn_valid=(xtrn, xval, ytrn, yval))


def main():
    # initial setup
    st.set_page_config(page_title="Bulldozers", layout="wide")

    st.title("Bulldozers Auction Price Predicting")
    stage = st.sidebar.selectbox("Select Stage", options=STAGES)

    # load data
    data = _prepare_data()
    raw_train, raw_valid = data["raw"]
    xtrn, xval, ytrn, yval = data["trn_valid"]

    # match page/stage
    if stage == "Introduction":
        stages.intro.run(raw_train, raw_valid)
    elif stage == "EDA":
        stages.eda.run(raw_train, raw_valid)
    elif stage == "Modeling":
        stages.models.run((xtrn, ytrn), (xval, yval))


if __name__ == "__main__":
    main()
