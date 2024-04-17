import numpy as np
import pandas as pd
from typing import List
from sklearn.preprocessing import QuantileTransformer
from sklearn.preprocessing import RobustScaler
from sklearn.preprocessing import MinMaxScaler
from sklearn import preprocessing as skpp

from modules.log import logger
from modules.preprocessing.pre_processing import PreProcessing


class ScalePreProcessing(PreProcessing):
    """
    Scaling.

    Uses different scaling preprocessings, mostly from SciKit-Learn.

    - If 'model_class' is not provided and 'model_class' key is not found in kwargs, RobustScaler
      is used as the default scaler model.
    - This class supports various scaler types, including QuantileTransformer, RobustScaler,
      and MinMaxScaler, with different distribution options.
    - Custom scaler classes from scikit-learn can be provided as strings and instantiated dynamically.
      If the provided model_class is not recognized or cannot be instantiated, a warning is logged, and
      RobustScaler is used as the default scaler model.
    """
    def __int__(self, columns, model_class, **kwargs):
        """
        :param columns: Columns to be preprocessed with it.
            (list)
        :param model_class: Class of the actual scaler model. - optional
            (str, type or None)
        :param kwargs: Additional parameters for the scaler model initialization. - optional
            (any, multiple -> dict)
        :return: None
        """
        super().__init__(columns)
        model = self.parse_model(model_class, kwargs)
        logger.info(f'Use {model.__str__()}!')
        self.model = model
        self.fitted = False

    @staticmethod
    def parse_model(model_class, kwargs):

        """
        Parse and instantiate a scaler model based on the provided model class and keyword arguments.

        Parses the model class and keyword arguments to instantiate a scaler model
        It supports various scaler types including QuantileTransformer, RobustScaler
        and MinMaxScaler, along with custom scaler classes.

        NOTE: Those classes have then to function in a similar fashion than any SciKit
                preprocessing! - Which means they should contain:
                1. `fit_transform` - taking the data *x* (ndarray or DatFrame) as input.
                2. `transform` - taking the data *x* (ndarray or DatFrame) as input.

        The steps, it parses the scaler are the following:

        1. The code checks if the word 'quantile' is in the model_class string.
        2. It extracts the distribution type by removing the substring 'quantile' and
            additional modifiers like 'transform' or 'preprocessing'.
        3. If a valid distribution type ('uniform' or 'normal') is detected, it
            instantiates a QuantileTransformer with the specified distribution using the
            `output_distribution` parameter.
        4. If no valid distribution type is found or if 'quantile' is not part of
            `model_class`, it falls back to instantiating a default
            `QuantileTransformer`.

        Example Usage:
        >>> from sklearn.preprocessing import RobustScaler
        >>> kwargs = {'quantile_transformer': True, 'output_distribution': 'uniform'}
        >>> model = ScalePreProcessing.parse_model('quantile', kwargs)
        >>> isinstance(model, QuantileTransformer)
        True

        >>> model = ScalePreProcessing.parse_model(RobustScaler, {})
        >>> isinstance(model, RobustScaler)
        True

        :param model_class: The class or name of the scaler model to be instantiated. If None, the function
                       attempts to retrieve the model class from the provided keyword arguments using
                       the 'model_class' key. If a string is provided, it is case-insensitively matched
                       against supported scaler types.
                       (Union[type, str, None])

        :param kwargs: Additional keyword arguments used for configuring the scaler model during instantiation.
                  (dict)

        :return: An instance of the scaler model configured with the provided keyword arguments.
                 (object)
        """
        if isinstance(model_class, type):
            model_class = model_class.__name__
        if model_class is None:
            model_class = kwargs.get('model_class', RobustScaler)
        if isinstance(model_class, str):
            model_class = model_class.lower()
            if 'quantile' in model_class:
                dist = model_class.replace('quantile', '')
                if 'transform' in model_class:
                    dist = dist.replace('transform', '')
                if 'preprocessing' in model_class:
                    dist = dist.replace('preprocessing', '')
                if len(dist) > 0 and dist in ['uniform', 'normal']:
                    if "output_distribution" in kwargs:
                        del kwargs["output_distribution"]
                    model = QuantileTransformer(output_distribution=dist, **kwargs)
                else:
                    model = QuantileTransformer(**kwargs)
            elif 'robust' in model_class:
                model = RobustScaler(**kwargs)
            elif 'minmax' in model_class or 'maxmin' in model_class:
                model = MinMaxScaler(**kwargs)
            else:
                if hasattr(skpp, model_class):
                    model = getattr(skpp, model_class)(**kwargs)
                else:
                    logger.warn(f'Cannot parse scaler class {model_class}!')
                    model = RobustScaler(**kwargs)
        elif not isinstance(model_class, type):
            model = model_class
        else:
            model = model_class(**kwargs)
        return model

    def transform(self, df: pd.DataFrame):
        if not self.fitted:
            self.model = self.model.fit(df[self.columns].values)
            self.fitted = True
        df[self.columns] = self.model.transform(df[self.columns].values)
        return df


class NormalQuantilePreProcessing(ScalePreProcessing):
    """
    Preprocessing wrapper for QualtileTransformer using Normal distribution.
    """
    def __init__(self, columns: List, **kwargs):
        if "output_distribution" in kwargs:
            del kwargs["output_distribution"]
        model = QuantileTransformer(output_distribution="normal", **kwargs)
        super().__init__(columns, model_class=model)
        self.fitted = False
        self.model = model


class UniformQuantilePreProcessing(ScalePreProcessing):
    """ Preprocessing wrapper for QualtileTransformer using uniform distribution. """
    def __init__(self, columns, **kwargs):
        if "output_distribution" in kwargs:
            del kwargs["output_distribution"]
        model = QuantileTransformer(output_distribution="uniform", **kwargs)
        super().__init__(columns, model_class=model)
        self.fitted = False
        self.model = model


class RobustScalerPreProcessing(ScalePreProcessing):
    """ Preprocessing wrapper for RobustScaler. """
    def __init__(self, columns, **kwargs):
        if "distribution" in kwargs:
            del kwargs["distribution"]
        model = RobustScaler(**kwargs)
        super().__init__(columns, model_class=model)
        self.fitted = False
        self.model = model


class LogScalePreProcessing(PreProcessing):
    """
    Log-Scale Preprocessing

    It transforms the incomming DataFrame values of `columns` as follows:

    :math:`$X^* = sign(X) ln(1 + |X|)$`

    The additional *1* is used to prevent NaNs, as well as the sign preservance.

    :param columns: list of columns to be transformed

    """
    def __init__(self, columns):
        super().__init__(columns)

    def transform(self, df: pd.DataFrame):
        sign = df[self.columns].values.sign()
        df[self.columns] = sign * np.log1p(df[self.columns].values.abs())
        return df
