import logging
from inspect import getfullargspec
from typing import Any, Callable, Sequence

import numpy as np
import pandas as pd
from sklearn.exceptions import NotFittedError
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

from .helper import create_logger
from .typehint import NumberSequence


def rmse_score(y_true, y_pred):
    return mean_squared_error(y_true, y_pred) ** 0.5


EVAL_LOGGER = create_logger("evaluate", "evalutate.log", level=logging.INFO)
DEFAULT_METRICS = [rmse_score, mean_absolute_error, r2_score]


def evalutate(
    estimator: Any,
    xval: pd.DataFrame,
    yval: pd.Series,
    eval_fns: list[Callable[..., float]] = DEFAULT_METRICS,
    round_ndigits: int = 5,
) -> np.ndarray:
    try:
        yprd = estimator.predict(xval.values)
    except NotFittedError as err:
        EVAL_LOGGER.error(repr(err))
        raise err
    else:
        scores = np.empty_like(eval_fns, dtype=float)
        for idx, fn in enumerate(eval_fns):
            arg_specs = getfullargspec(fn).args
            if arg_specs != ["y_true", "y_pred"]:
                err = ValueError(
                    f"{fn.__name__} does not contain argument specifications: {arg_specs}"  # NOQA
                )
                EVAL_LOGGER.error(repr(err))
                raise err

            scores[idx] = round(fn(yval.values, yprd), round_ndigits)
    return scores


def feature_importances(
    features: Sequence[str],
    importance: NumberSequence,
    threshold: float = 1.0,
) -> pd.DataFrame:
    assert 0 <= threshold < 1

    fi = pd.DataFrame(dict(features=features, importance=importance))
    fi.set_index("features", inplace=True)
    fi.sort_values(by="importance", ascending=False, inplace=True)

    fi["rank"] = np.arange(1, fi.shape[0] + 1)
    fi["norm_cummulative"] = fi.importance.cumsum() / fi.importance.sum()
    fi = fi[fi.norm_cummulative <= threshold]
    fi.columns = ["Importance", "Rank", "Normalized Cummulative"]

    return fi
