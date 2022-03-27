import utils as ut
import pandas as pd
import streamlit as st
import plotly.graph_objects as go

from dirs import DATA_DIR


@st.experimental_singleton(show_spinner=False)
def _load_data_dict() -> pd.DataFrame:
    data_dict = pd.read_feather(DATA_DIR / "data_dict.feather")
    data_dict.Variable.replace(
        {
            "ProductClassDesc": "fiProductClassDesc",
            "Saledate": "saledate",
            "Saleprice": "SalePrice",
            "State": "state",
            "Tip_control": "Tip_Control",
        },
        inplace=True,
    )

    data_dict.set_index("Variable", inplace=True)
    return data_dict


def run(dataframe: pd.DataFrame) -> None:
    st.header("Introduction")

    overview = st.container()
    data = st.container()

    # 1. overview
    with overview:
        st.subheader("1. Overview")
        ut.display_content("overview.txt")

    # 2. data
    with data:
        st.subheader("2. Data")
        ut.display_header("A look at the data", 4)
        st.markdown("The first 20 samples:")
        st.caption(
            "__NOTE__: Hover on the feature name to see its full name."
        )

        # load data description
        data_dict = _load_data_dict()
        ut.styleit(st.dataframe)(dataframe.head(20), height=1000)

        st.write(
            f"Shape:\n\nTraining Set:`{dataframe.shape}`\n\nValidation Set:`{(11573, 50)}`" # NOQA
        )

        # drop-down box for selecting feature to display description
        ut.display_header("Feature Description", 4)
        feat = st.selectbox("Select feature", options=data_dict.index)
        desc = data_dict.loc[feat].values[0].strip().capitalize()
        ut.display_header(f"Description:\n{desc}", 6)

        # display summary of the chosen feature
        try:
            x = dataframe[feat]
        except KeyError:
            pass
        else:
            info = x.describe().to_dict()
            info["na count"] = x.isna().sum()
            info["data type"] = x.dtype

            ut.display_header("Summary: ", 6)

            summary_col, plot_col = st.columns(spec=[1, 3])
            with summary_col:
                for k, v in info.items():
                    st.write(f"{k.capitalize()}:`{v}`")

            with plot_col:
                histogram = go.Histogram(x=x, nbinsx=45)
                ut.render(
                    go.Figure(data=histogram),
                    layout=dict(
                        height=525,
                        xaxis_title=feat,
                        yaxis_title="Count",
                    )
                )

        st.warning(
            "__WARNING__: The data is its raw form. The graph and summary statistics of some variables may not be reasonable." # NOQA
        )
