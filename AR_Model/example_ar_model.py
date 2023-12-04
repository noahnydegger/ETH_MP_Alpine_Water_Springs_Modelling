"""
    Example on how to use AutoReg Class from statsmodels.

    When learning a new modeling tool, *always* use data from a perfectly known model.
"""

# Copyright (C) 2023 OST Ostschweizer Fachhochschule
#
# This program is free software; you can redistribute it and/or modify
# it under the terms of the GNU General Public License as published by
# the Free Software Foundation; either version 3 of the License, or
# (at your option) any later version.
#
# This program is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
# GNU General Public License for more details.
#
# You should have received a copy of the GNU General Public License
# along with this program. If not, see <http://www.gnu.org/licenses/>.

# Author: Juan Pablo Carbajal <juanpablo.carbajal@ost.ch>
import numpy as np
from numpy.typing import ArrayLike
import matplotlib.pyplot as plt
from statsmodels.tsa.ar_model import AutoReg, AutoRegResults

rng = np.random.default_rng()


def create_data(*, weights: ArrayLike, lags: ArrayLike,
                samples: int, x0: ArrayLike) -> np.ndarray:
    """Create autoregressive data.

        Parameters
        ----------
        lags,weights:
            Lagas and weigths of the model. Lag caanot be 0.
        x0:
            Initial state. Need as many as the maximum lag.
        samples:
            Number of samples to generate

        Returns
        -------
        data:
            Generated data
    """
    weights, lags, x0 = [np.asarray(u) for u in (weights, lags, x0)]
    data = np.zeros(samples)
    order = lags.max()
    data[:order] = x0
    # make sure lags are sorted
    lags = np.sort(lags)[::-1]
    for n in range(order, samples-order):
        delayed = n - lags
        data[n] = data[delayed] @ weights
    return data


def sample_ar_path(res: AutoRegResults, *, xinit: np.ndarray, steps: int) -> np.ndarray:
    """Sample a path from a fitted AR model.
    """
    coefs = res.params
    if xinit.size < coefs.size - 1:
        raise ValueError(f"Not enough initial values to start: {xinit.size} < {coefs.size}")
    noise = rng.normal(loc=0, scale=np.sqrt(res.scale), size=steps-xinit.size)
    lags = np.asarray(res.ar_lags)
    path = np.zeros(steps)
    path[:xinit.size] = xinit.copy()
    for i in range(xinit.size, steps):
        path[i] = path[i - lags] @ coefs[1:] + coefs[0] + noise[i-xinit.size]
    return path


if __name__ == "__main__":
    # Create data
    l = np.array([1, 3, 4, 5])
    w = np.array([0.5, -0.25, 0.125, -0.1])
    # need as many initial values as the highest lag
    x0 = np.array([0, 0.1, 0.2, -0.1, 0.3])
    data_latent = create_data(weights=w, lags=l, x0=x0, samples=100)
    # add noise
    factor = 0.1  # TODO: set this to different values
    data = data_latent + rng.normal(loc=0.0, scale=factor * np.abs(data_latent).max(),
                                    size=data_latent.shape)
    t = np.arange(data.size)

    # split data
    trn_end = 25
    trn = data[:trn_end]
    tst = data[trn_end:]

    # Fit using statsmodel
    mod = AutoReg(trn, l, old_names=False)
    res = mod.fit()
    print(res.summary())

    data_ = res.predict(end=data.size-1, dynamic=trn_end)

    # Sample some paths
    Np = 25
    paths = np.zeros((Np, data.size))
    for n in range(Np):
        paths[n] = sample_ar_path(res, xinit=x0, steps=data.size)
    path_args = dict(color='r', linestyle='-', alpha=0.1, label="model paths")
    fig, ax = plt.subplots()
    ax.plot(t, paths.T, **path_args)
    ax.plot(t, data_latent, "-", label="latent")
    ax.plot(t[:trn_end], trn, "o", label="training")
    ax.plot(t[trn_end:], tst, "x", label="testing")
    ax.plot(t, data_, "b--", label="model")
    hdl, lbl = ax.get_legend_handles_labels()
    plt.legend(hdl[n:], lbl[n:])

    # using integrated plot function
    fig = res.plot_predict(start=0, end=data.size-1)
    ax = plt.gca()
    ax.plot(t, data_latent, ":", label="latent")
    ax.plot(t[:trn_end], trn, "o", label="training")
    ax.plot(t[trn_end:], tst, "x", label="testing")
    ax.plot(t, paths.T, **path_args)
    hdl, lbl = ax.get_legend_handles_labels()
    plt.legend(hdl[:5], lbl[:5])

    plt.show()
