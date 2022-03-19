import utils as ut
import pandas as pd
import streamlit as st
import plotly.graph_objects as go

from typing import Any

from dirs import CONTENT_DIR
from stages.modeling.configuration.config import RandomForestSetup
from stages.modeling.configuration.config import ExtraTreesSetup
from stages.modeling.configuration.config import GradientBoostingSetup

MODEL_LOGGER = ut.create_logger(name="models", logfile="fit.log")
MODELS = {
    "Random Forest": RandomForestSetup,
    "Extra Trees": ExtraTreesSetup,
    "Gradient Boost": GradientBoostingSetup,
}


@st.experimental_singleton(show_spinner=False)
@ut.loggit(logger=MODEL_LOGGER)
@ut.timeit
def fit(estimator, params: dict[str, Any], X: pd.DataFrame, y: pd.Series):
    return estimator(**params).fit(X.values, y.values)


def run(
    train: tuple[pd.DataFrame, pd.Series],
    valid: tuple[pd.DataFrame, pd.Series],
) -> None:

    xtrn, ytrn = train
    xval, yval = valid

    for key in ["rmse", "mae", "r2"]:
        if key not in st.session_state:
            st.session_state[key] = 0

    st.header("Modeling")

    configs = st.container()
    metrics = st.container()
    featimp = st.container()

    with configs:

        st.subheader("1. Configurations")
        model_name = st.selectbox("Select Model", options=MODELS.keys())
        # handling options
        model_config = MODELS[model_name]()
        estimator, hyperparams = model_config()
        # handling button
        clicked = st.button("Click to fit")

        if clicked:
            with st.spinner("Training..."):
                estimator, time_taken = fit(
                    estimator, hyperparams, xtrn, ytrn
                )

            st.success(f"Finished in {round(time_taken, 3)} second(s).")

            # Metrics
            with metrics:
                st.subheader("2. Metrics")
                cols = st.columns(3)
                rmse, mae, r2 = ut.evalutate(estimator, xval, yval)

                with cols[0]:
                    ut.display_metric(
                        rmse,
                        "RMSE - Root Squared Mean Error",
                        "rmse",
                        smaller_better=True,
                    )
                with cols[1]:
                    ut.display_metric(
                        mae,
                        "MAE - Mean Absolute Error",
                        "mae",
                        smaller_better=True,
                    )
                with cols[2]:
                    ut.display_metric(
                        r2,
                        "R2 - R-squared",
                        "r2",
                        help="The maximum value is 1.",
                    )

            # Feature Importances
            with featimp:
                st.subheader("3. Feature Importances")

                fi = ut.feature_importances(
                    features=xtrn.columns,
                    importance=estimator.feature_importances_,
                    threshold=0.95,
                )

                fi_styled = fi.style.background_gradient(
                    cmap=ut.CM_BLUE,
                    subset=["Importance", "Normalized Cummulative"],
                )

                # table
                ut.styleit(st.table)(data=fi_styled)

                # graph
                x, y = fi.Importance, fi.index
                barchart = go.Bar(
                    x=x,
                    y=y,
                    orientation="h",
                    text=x,
                    texttemplate="%{x:.4f}",
                    marker_color=fi.Importance,
                )

                ut.render(
                    go.Figure(data=barchart),
                    layout=dict(
                        colorscale_sequential=ut.cmap2hexlist(),
                        xaxis_title="Importance",
                        yaxis_title="Variables",
                    )
                )

                with st.expander("NOTE: "):
                    ut.display_content(CONTENT_DIR / "featimp-note.txt")
