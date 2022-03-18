import pandas as pd
import numpy as np
from sklearn.preprocessing import robust_scale
from sklearn.ensemble import RandomForestRegressor, RandomForestClassifier
from sklearn.feature_selection import f_regression
from jax import lax, vmap, numpy as jnp
from tqdm.notebook import trange, tqdm


def pearsonr(x, y):
    lx, ly = x.shape[0], y.shape[0]
    if lx != ly:
        raise ValueError('x and y must have the same length.')
    if lx < 2 or ly < 2:
        raise ValueError('x and y must have length at least 2.')

    xm, ym = x - jnp.mean(x), y - jnp.mean(y)
    normxm, normym = jnp.linalg.norm(xm), jnp.linalg.norm(ym)

    r = jnp.dot(xm/normxm, ym/normym)
    return lax.cond(jnp.isnan(r), lambda _: 1e-7, lambda x: x, r)


def mrmr(X, y, K, objective, method='rfcq', kw=None):
    assert isinstance(
        X, pd.DataFrame), "`X` must be an instance of pandas.DataFrame."
    assert objective in [
        'regression', 'classification'], 'Supported values for `objective` are "regression" and "classification".'
    assert method in [
        'rfcq', 'fcq'], 'Supported values for `method` are "rfcq" and "fcq".'

    x = pd.DataFrame(robust_scale(X), columns=X.columns, dtype=np.float32)

    if method == 'fcq':
        relevance = np.nan_to_num(f_regression(x, y)[0], 0)
    if method == 'rfcq':
        if kw is None:
            kw = dict(max_depth=10, min_samples_leaf=50,
                      n_estimators=50, max_samples=.15)
        rf = RandomForestRegressor(
            **kw) if objective == 'regression' else RandomForestClassifier(**kw)
        rf.fit(x, y)
        relevance = rf.feature_importances_

    first = np.argmax(np.abs(relevance))
    relevance[first] = 0
    selected = [x.columns[first]]

    for i in trange(K-1):
        corr = vmap(vmap(pearsonr, (1, None)), (None, 1))(
            x.values, x[selected].values)
        coef = relevance/jnp.abs(corr).mean(axis=0)
        toadd_idx = coef.argmax()
        relevance[toadd_idx] = 0
        selected.append(x.columns[toadd_idx])
    return np.asarray(selected)
