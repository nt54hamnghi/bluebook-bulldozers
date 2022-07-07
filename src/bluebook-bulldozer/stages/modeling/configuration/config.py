import pathlib
from typing import Any

import streamlit as st
from sklearn.ensemble import (
    ExtraTreesRegressor,
    GradientBoostingRegressor,
    RandomForestRegressor,
)

import utils as ut

CURR_DIR = pathlib.Path(__file__).parent
JSON_DIR = CURR_DIR / "config-json"


class BaseSetup:
    config_file: str

    def __call__(self) -> dict[str, Any]:
        self.container_configs = ut.read_json(
            json_name=JSON_DIR / self.config_file, as_namespace=True
        )
        description = self.container_configs.description
        configurable_prms = self.container_configs.params
        fixed_prms = self.container_configs.default

        with st.expander("See explanation"):
            st.markdown(description)

        self.cols = st.columns(2)

        for param, value in configurable_prms.items():
            kwds = value["args"]
            which_col = value["column"]
            widget = value["widget"]

            if widget == "slider":
                fixed_prms[param] = self.cols[which_col].slider(**kwds)
            elif widget == "radio":
                fixed_prms[param] = self.cols[which_col].radio(**kwds)

        return fixed_prms


class RandomForestSetup(BaseSetup):
    config_file = "random-forest.json"

    def __call__(self) -> tuple[RandomForestRegressor, dict[str, Any]]:
        fixed_prms = super().__call__()
        fixed_prms["max_depth"] = (
            None if fixed_prms["max_depth"] == 0 else fixed_prms["max_depth"]
        )
        return RandomForestRegressor, fixed_prms


class ExtraTreesSetup(BaseSetup):
    config_file = "extra-trees.json"

    def __call__(self) -> tuple[ExtraTreesRegressor, dict[str, Any]]:
        fixed_prms = super().__call__()

        fixed_prms["max_depth"] = (
            None if fixed_prms["max_depth"] == 0 else fixed_prms["max_depth"]
        )

        if fixed_prms["bootstrap"]:
            max_samples = self.container_configs.params["max_samples"]
            fixed_prms["max_samples"] = self.cols[
                max_samples["column"]
            ].slider(**max_samples["args"])

        return ExtraTreesRegressor, fixed_prms


class GradientBoostingSetup(BaseSetup):
    config_file = "gradient-boost.json"

    def __call__(self) -> tuple[GradientBoostingRegressor, dict[str, Any]]:
        fixed_prms = super().__call__()

        if fixed_prms["early_stopping"]:
            n_iter_no_change = self.container_configs.params[
                "n_iter_no_change"
            ]
            fixed_prms["n_iter_no_change"] = self.cols[
                n_iter_no_change["column"]
            ].slider(**n_iter_no_change["args"])

        fixed_prms.pop("early_stopping")

        st.caption(
            "Due to the sequential nature, Gradient Boost's training time is relatively long."  # noqa
            + " For 200 iterations, it takes around 120 to 180 seconds to complete."  # noqa
            + " You can decrease the iteration count to make it faster; however, you should also increase the learning rate to achieve a comparable accuracy."  # noqa
        )

        return GradientBoostingRegressor, fixed_prms
