import numpy as np
import pandas as pd
from collections import Counter
from tqdm.notebook import trange, tqdm
from sklearn.tree import BaseDecisionTree
from sklearn.ensemble import BaseEnsemble, ExtraTreesRegressor


def _create_shadow(df, noise=None, random_state=None):
    """
    Take in as input a DataFrame containing x number of features. 
    For each feature, create a copy, shuffle it, and add it into the DataFrame.
    """
    assert isinstance(
        df, pd.DataFrame), "Argument `df` must be an instance of pandas.DataFrame"
    if isinstance(random_state, int):
        np.random.seed(random_state)
    cp = df.apply(np.random.permutation).copy()

    if noise is not None:
        assert noise >= 0, "Noise Power must be a positive number, in range (0, inf)"
        def add_noise(x, noise): return np.random.normal(
            x.mean() + noise, noise, x.size) + x
        cp = cp.apply(add_noise, args=(noise,))

    cp.columns = "shadow_" + cp.columns
    new = df.merge(cp, left_index=True, right_index=True)

    return new


def hiwamari(estimator, X, y, n_iters=100, max_samples=1, early_stopping_rounds=None, scale_factor=1, noise=None):
    assert isinstance(estimator, (BaseDecisionTree, BaseEnsemble)
                      ), "Hiwamari supports only sklearn.tree and sklearn.ensemble models."
    assert 0 < scale_factor <= 1, "scale_factor must be in range (0,1]"
    # assert hasattr(estimator, "feature_importances_")

    tmp = _create_shadow(X, noise)
    cols = tmp.columns.to_numpy()
    counter = Counter()
    kw = dict(n=max_samples) if isinstance(
        max_samples, int) else dict(frac=max_samples)

    for i in trange(n_iters):
        x = tmp[cols].sample(**kw)
        estimator.fit(x, y[x.index])
        fi = estimator.feature_importances_

        light, shadow = np.split(fi, 2)
        keep = light >= (shadow * scale_factor)

        cols = cols[np.tile(keep, reps=2)]

        if early_stopping_rounds is not None:
            counter.update([keep.sum()])
            if(counter.most_common()[0][1] > early_stopping_rounds):
                print(f"Stopping early at {i}")
                break

    return cols[:len(cols)//2]
