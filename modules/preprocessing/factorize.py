from typing import Iterable

import numpy as np
import pandas as pd
from modules.preprocessing import PreProcessing


class Factorize(PreProcessing):
    """
    Weighted Factorization

    Here, we factorize a feature NxM matrix with a 1xM factor model.
    The factors are declared by a fixed weighting.
    Additionally, a bias is declared.
    The bias is usually used as a buffer for the factorization and is only added
    afterward, iff it is 1.0 or higher!

    Thus:

        :math:`$y^* = y (f_f(x_f) + B_f)$`

    if `normalization` is off, else:

         :math:`$y^* = y ((f_f(x_f) + B_f) / Z)$`

         with :math:`$Z = \sum(f_f(x_f) + B_f)$

    here, :math:`$x_f$`` is the set of features corresponding to factorization *f*,
    :math:`$f_f$` is the actual factorization function for the set and :math:`$B_f$` is
    the actual bias across the entire function..

    :param columns: Columns to be factorized
    :param factors: Factors (need to be the same length than `columns` or length 1)
    :param bias: Bias, a single scalar to be either the buffer or real bias.
    :param normalize_factors: Whether to normalize factors.
    :param bias_in_factors: Whether to use bias as buffer.
    :param default_on_missing:
    """
    def __init__(
            self,
            columns: list,
            factors: list,
            bias: float = None,
            normalize_factors: bool = True,
            bias_in_factors: bool = True,
            default_on_missing: float = 0,
            suffix: str = '_wfac'
    ):
        super().__init__(columns)
        if not isinstance(factors, Iterable):
            factors = [factors]
        assert (len(factors) == len(columns)) or (len(factors) == 1)
        if len(factors) == 1:
            factors = np.ones(len(columns)) * factors
        factors = np.array(factors, dtype=np.float64)
        if normalize_factors:
            factors /= np.sum(factors)
        if bias is None:
            bias = 0
        if bias_in_factors and np.abs(bias) < 1.0:
            factors *= (1 - bias)
            factors += bias
            bias = 0
        self.factors = factors
        self.bias = bias if bias is not None else 0
        self.default_on_missing = default_on_missing
        self.new_cols = (pd.Series(columns) + suffix).tolist()

    def inverse_transform(self, df):
        columns = self.columns
        factors = self.factors
        bias = self.bias
        df[columns] /= factors[None, :]
        df[columns] -= bias
        return df

    def transform(self, df: pd.DataFrame) -> pd.DataFrame:
        columns = self.columns
        df[
            list(filter(lambda c: c not in df.columns, columns))
        ] = self.default_on_missing
        factors = self.factors
        bias = self.bias
        columns = self.columns
        df[columns] *= factors[None, :]
        df[columns] += bias
        return df


class UniFactorize(Factorize):
    """
    Uniorm factorization.

    For a set of features, their influence shall be quantized uniforml.
    This is, in particular, important, if there is little knowledge and they are not
    singleton (only one active at a time)!

    :math:`$x_{uni} = x / L_x + B_f$`

    If `normalize_factors` is *true*:

    :math:`$x_{uni} = (x / L_x + B_f) / Z$`

        with :math:`$Z = \sum(x / L_x + B_f)$`

    with :math:`$B_f$` the bias, the considered feature set *x* and :math:`$L_x$` the
    length of this feature set.

    :param columns: the considered feature set
    :param bias: the bias
    :param normalize_factors: whether to normalize the final result
    :param bias_in_factors: whether the bias is inside the factor function
    :param default_on_missing: a defualt value for missing values.
    """
    def __init__(
        self,
        columns: list,
        bias: float = 0,
        normalize_factors: bool = True,
        bias_in_factors: bool = True,
        default_on_missing: float = 0
    ):
        super().__init__(
            columns,
            1/len(columns),
            bias,
            normalize_factors,
            bias_in_factors,
            default_on_missing,
            '_uni'
        )
