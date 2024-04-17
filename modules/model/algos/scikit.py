# %% imports
import numpy as np
from sklearn.base import BaseEstimator
from sklearn.linear_model import PoissonRegressor
from skopt import gp_minimize
from skopt.space import Categorical, Real
from functools import partial


# %% main classes
class ADBUDGRegressor:
    r"""
    Run a single ADBUDG regression model.

    This model uses a GP-minimizer to tune the non-linear parameters of an ADBUDG
    response curve, which looks as follows:

    ..math::
        ADBUDG(x_i) = \\frac{x_^{\\gamma_i}}{\\rho_i + x^{\\gamma_i}}

    The non-linear parameters (`rho` and `gamma`)
    respectively can be either tuned with a discrete search or a continuous search.
    For discrete, `d_gam` and `d_rho` have to be set and `discrete_search=True`.

    @param n_input: no. inputs for the regression
    @param regressor_class: SciKit-Learn regressor class
    @param gam_range: lower and higher bound for al gamma search spaces
    @param rho_range: lower and higher bound for al rho search spaces
    @param discrete_search: if using discrete search spaces
    @param d_gam: discrete step size for al gamma search spaces
    @param d_rho: discrete step size for al rho search spaces
    @param model_params: parameters for the SciKit regression model
    """
    def __init__(
            self,
            n_input: int,
            regressor_class: BaseEstimator = PoissonRegressor,
            gam_range: tuple = (0.1, 3),
            rho_range: tuple = (0.1, 2),
            discrete_search: bool = True,
            d_gam: float = .1,
            d_rho: float = .1,
            model_params: dict = None
    ):
        if model_params is None:
            model_params = dict(
                alpha=.8,
                solver='newton-cholesky',
                max_iter=500
            )
        self.model = regressor_class(**model_params)  # noqa
        self.regressor_class = regressor_class
        self.model_params = model_params
        self.D = n_input
        self.gam_range = gam_range
        self.rho_range = rho_range
        self.gammas = np.ones(n_input)
        self.rhos = np.ones(n_input)
        self.base = 1
        self.discrete_search = discrete_search
        self.d_gam = d_gam
        self.d_rho = d_rho

    def get_params(self, deep=None):  # noqa
        if deep:
            self.deep = True
        return dict(
            regressor_class=self.regressor_class,
            model_params=self.model_params,
            n_input=self.D,
            gam_range=self.gam_range,
            rho_range=self.rho_range,
            discrete_search=self.discrete_search,
            d_gam=self.d_gam,
            d_rho=self.d_rho
        )

    @staticmethod
    def transform_vars(x, gammas, rhos):
        x_pow = np.sign(x) * np.abs(x)**gammas
        return x_pow / (x_pow + rhos)

    def opt_params(self, params, x, y):

        x = np.array(x).astype(float)
        y = np.array(y).astype(float)

        D = self.D  # noqa
        gammas = np.array(params[:D]).astype(float)
        rhos = np.array(params[-D:]).astype(float)

        x = self.transform_vars(x, gammas, rhos)
        x[np.isnan(x)] = 1e-8

        model = self.model
        return 1 - model.fit(x, y).score(x, y)  # noqa

    def fit(self, x, y):

        x = np.array(x).astype(float)
        y = np.array(y).astype(float)

        self.base = np.min(x) + 1e-8

        if not self.discrete_search:
            gam_spaces = [Real(*self.gam_range)] * self.D
            rho_spaces = [Real(*self.rho_range)] * self.D
        else:
            gam_spaces = [Categorical(
                np.round(np.arange(
                    self.gam_range[0],
                    self.gam_range[1] + self.d_gam,
                    self.d_gam
                ), 4)
            )] * self.D
            rho_spaces = [Categorical(
                np.round(np.arange(
                    self.rho_range[0],
                    self.rho_range[1] + self.d_rho,
                    self.d_rho
                ), 4)
            )] * self.D

        params = gp_minimize(
            partial(self.opt_params, x=x, y=y),
            gam_spaces + rho_spaces,
            n_calls=450,
            n_points=600,
            n_jobs=-1
        ).x
        gammas = np.array(params[:self.D])
        rhos = np.array(params[-self.D:])

        self.gammas = gammas
        self.rhos = rhos

        x = self.transform_vars(x, self.gammas, self.rhos)
        x[np.isnan(x)] = 1e-8

        self.model = self.model.fit(x, y)

        return self

    def score(self, x, y):

        x = np.array(x).astype(float)
        y = np.array(y).astype(float)

        x = self.transform_vars(x, self.gammas, self.rhos)
        x[np.isnan(x)] = 1e-8

        return self.model.score(x, y)

    def predict(self, x):

        x = np.array(x).astype(float)

        x = self.transform_vars(x, self.gammas, self.rhos)
        x[np.isnan(x)] = 1e-8

        return self.model.predict(x)


class LogNormalRegressor:
    """
    Run a single Log-Normal regression model.

    This model uses a GP-minimizer to tune the non-linear parameters of an log-normal
    response curve, which looks as follows:

    ..math::
        LogNormal(x_i) = \\frac{1}{x_i\\sqrt{\\sigma^2 2\\pi}} \exp{-\\frac{(ln(x_i - \\mu)^2}{2\\sigma^2}}

    The non-linear parameters (`mue` and `std`)
    respectively can be either tuned with a discrete search or a continuous search.
    For discrete, `d_std` and `d_mue` have to be set and `discrete_search=True`.

    @param n_input: no. inputs for the regression
    @param regressor_class: SciKit-Learn regressor class
    @param std_range: lower and higher bound for al gamma search spaces
    @param mue_range: lower and higher bound for al rho search spaces
    @param discrete_search: if using discrete search spaces
    @param d_std: discrete step size for al scale search spaces
    @param d_mue: discrete step size for al mean search spaces
    @param model_params: parameters for the SciKit regression model
    """  # noqa
    def __init__(
        self,
        n_input: int,
        regressor_class: BaseEstimator = PoissonRegressor,
        std_range: tuple = (0.1, 3),
        mue_range: tuple = (0.1, 2),
        discrete_search: bool = True,
        d_std: float = .1,
        d_mue: float = .1,
        n_calls: int = 245,
        n_points: int = 500,
        model_params: dict = None
    ):
        if model_params is None:
            model_params = dict(
                alpha=.8,
                solver='newton-cholesky',
                max_iter=500
            )
        self.model = regressor_class(**model_params)  # noqa
        self.regressor_class = regressor_class
        self.model_params = model_params
        self.D = n_input
        self.std_range = std_range
        self.mue_range = mue_range
        self.stds = np.ones(n_input)
        self.mues = np.ones(n_input)
        self.base = 1
        self.n_calls = n_calls
        self.n_points = n_points
        self.discrete_search = discrete_search
        self.d_std = d_std
        self.d_mue = d_mue

    def get_params(self, deep=None):  # noqa
        if deep:
            self.deep = True
        return dict(
            regressor_class=self.regressor_class,
            model_params=self.model_params,
            n_input=self.D,
            std_range=self.std_range,
            mue_range=self.mue_range,
            discrete_search=self.discrete_search,
            d_std=self.d_std,
            d_mue=self.d_mue
        )

    @staticmethod
    def transform_vars(x, std, mue):
        x += np.min(x) + 1e-8
        return (1/(x * np.sqrt(std**2*2*np.pi)))*np.exp(-(np.log(x) - mue)**2/(2*std**2))  # noqa

    def opt_params(self, params, x, y):

        x = np.array(x).astype(np.float64)
        y = np.array(y).astype(np.float64)

        D = self.D  # noqa
        stds = np.array(params[:D]).astype(float)
        mues = np.array(params[-D:]).astype(float)

        x = self.transform_vars(x, stds, mues)
        x[np.isnan(x)] = 1e-8

        model = self.model
        return 1 - model.fit(x, y).score(x, y)  # noqa

    def fit(self, x, y):

        x = np.array(x).astype(np.float64)
        y = np.array(y).astype(np.float64)

        self.base = np.min(x) + 1e-8

        if not self.discrete_search:
            std_spaces = [Real(*self.std_range)] * self.D
            mue_spaces = [Real(*self.mue_range)] * self.D
        else:
            std_spaces = [Categorical(
                np.round(np.arange(
                    self.std_range[0],
                    self.std_range[1] + self.d_std,
                    self.d_std
                ), 4)
            )] * self.D
            mue_spaces = [Categorical(
                np.round(np.arange(
                    self.mue_range[0],
                    self.mue_range[1] + self.d_mue,
                    self.d_mue
                ), 4)
            )] * self.D

        params = gp_minimize(
            partial(self.opt_params, x=x, y=y),
            std_spaces + mue_spaces,
            n_calls=self.n_calls,
            n_points=self.n_points,
            n_jobs=-1
        ).x
        stds = np.array(params[:self.D])
        mues = np.array(params[-self.D:])

        self.stds = stds
        self.mues = mues

        x = self.transform_vars(x, self.stds, self.mues)
        x[np.isnan(x)] = 1e-8

        self.model = self.model.fit(x, y)

        return self

    def score(self, x, y):

        x = np.array(x).astype(np.float64)
        y = np.array(y).astype(np.float64)

        x = self.transform_vars(x, self.stds, self.mues)
        x[np.isnan(x)] = 1e-8

        return self.model.score(x, y)

    def predict(self, x):

        x = np.array(x).astype(np.float64)

        x = self.transform_vars(x, self.stds, self.mues)
        x[np.isnan(x)] = 1e-8

        return self.model.predict(x)
