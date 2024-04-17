import pandas as pd
import numpy as np
from typing import List, Union
from sklearn.multioutput import MultiOutputRegressor

from modules.model import ModelWrapper
from sklearn.svm import SVR
from sklearn.gaussian_process import GaussianProcessRegressor

from modules.common import NCPU


class SciKitRegressor(ModelWrapper):
    """ SciKit Regressor Model.

    params:
        feature_columns: list
            ... the columns based to do predictions of.
        target_columns: list
            ... the columns to predict.
        model_class: type
            ... a scikit-like model class to init with **kwargs
        **kwargs: dict
            ... **kwargs for the model class `__init__`

    """

    def __init__(
        self,
        feature_columns: List[str],
        target_columns: List[str],
        model_class: type,
        **kwargs,
    ):
        super().__init__(feature_columns, target_columns)
        if len(target_columns) > 1:
            self.model = MultiOutputRegressor(
                model_class(**kwargs),
                n_jobs=NCPU
            )
        else:
            if hasattr(model_class, 'n_jobs'):
                kwargs['n_jobs'] = NCPU
            self.model = model_class(**kwargs)

    def save_params(self, path="./model_params/params.pkl"):
        self.save_obj(self.model.get_params(deep=True), path)

    def fit(self, df: pd.DataFrame, **kwargs) -> "SciKitRegressor":
        """Fit underlying Scikit-like Regressor!

        in:
            df: DataFrame
                ... data (must contain `self.targets` and `self.features`)
            **kwargs: dict
                ... additional parameters for training.

        out:
            self: SciKitRegressor
        """
        df = df.fillna(0)
        self.mu_x = np.nanmean(df[self.features].values, axis=0)  # noqa
        self.va_x = np.nanvar(df[self.features].values, axis=0)  # noqa
        self.model = self.model.fit(
            df[self.features].values, df[self.targets].values, **kwargs
        )  # noqa
        self.std = np.sqrt(
            (df[self.targets].values - self.predict(df).values) ** 2
        ).mean()  # noqa
        return self

    def predict(self, df: pd.DataFrame) -> pd.DataFrame:
        """Predict.

        in:
            df: DataFrame
                ... data (must contain `self.features`)
        out:
            prediction: pd.DataFrame
                ... actual model estimation.
        """
        df = self. _check_features_(df)
        return pd.DataFrame(
            self.model.predict(df[self.features].values),
            columns=self.targets,
            index=df.index,
        )

    def __str__(self):
        return f"Wrapper of model type {type(self.model)}"


class ScikitMixtureRegressor(SciKitRegressor):

    def __init__(
        self,
        feature_columns: List[str],
        target_columns: List[str],
        model_class: type,
        model_args: dict,
        sub_models: list,
        sub_model_args: list = None,
        from_pretrained: bool = False
    ):
        super().__init__(
            feature_columns,
            target_columns,
            model_class,
            **model_args
        )

        # parsing sub-models
        self.sub_models = []
        if not from_pretrained:
            for k in range(len(sub_models)):
                model = sub_models[k]
                margs = {}
                if sub_model_args is not None:
                    if len(sub_model_args) > k:
                        margs = sub_model_args[k]
                self.sub_models.append(model(**margs))
        else:
            self.sub_models = sub_models

        # wrapping it up
        for k in range(len(sub_models)):
            if not isinstance(self.sub_models[k], ModelWrapper):
                wrapper = SciKitRegressor(
                    feature_columns,
                    target_columns,
                    self.sub_models[k].__class__
                )
                wrapper.model = self.sub_models[k]
                self.sub_models[k] = wrapper

        # if trained
        self.from_pretrained = from_pretrained

    def fit_submodels(self, df: pd.DataFrame):
        if self.from_pretrained:
            return
        else:
            for k in range(len(self.sub_models)):
                self.sub_models[k] = self.sub_models[k].fit(df)

    def pred_submodels(self, df: pd.DataFrame) -> list:
        preds = []
        for k in range(len(self.sub_models)):
            preds.append(self.sub_models[k].predict(df).values)
        return np.concatenate(preds, axis=1)

    def fit(self, df: pd.DataFrame, **kwargs) -> "SciKitRegressor":
        self.fit_submodels(df)
        preds = self.pred_submodels(df)
        self.model = self.model.fit(preds, df[self.targets].values.reshape(len(df), -1))
        return self

    def predict(self, df: pd.DataFrame) -> pd.DataFrame:
        df = self. _check_features_(df)
        preds = self.pred_submodels(df)
        return pd.DataFrame(
            self.model.predict(preds),
            columns=self.targets
        )


class SupportVectorRegressorModel(SciKitRegressor):
    """ Support Vector Regressor Model.

    params:
        feature_columns: list
            ... the columns based to do predictions of.
        target_columns: list
            ... the columns to predict.
        model_class: type
            ... a scikit-like model class to init with **kwargs


    """

    def __init__(
        self,
        feature_columns: List[str],
        target_columns: List[str],
        kernel: str = "rbf",
        degree: int = 3,
        gamma: Union[str, float] = "scale",
        coef0: float = 0.0,
        tol: float = 0.001,
        C: float = 1.0,  # noqa
        epsilon: float = 0.1,
        shrinking: bool = True,
        cache_size: int = 200,
        verbose: bool = False,
        max_iter: int = -1,
    ):
        super().__init__(
            feature_columns,
            target_columns,
            SVR,
            kernel,  # noqa
            degree,  # noqa
            gamma,  # noqa
            coef0,  # noqa
            tol,  # noqa
            C,  # noqa
            epsilon,  # noqa
            shrinking,  # noqa
            cache_size,  # noqa
            verbose,  # noqa
            max_iter,  # noqa
        )

    def __str__(self):
        return f"Wrapper of model type {type(self.model)}"


class GaussianProcessModel(SciKitRegressor):
    """ Support Vector Regressor Model.

    params:
        feature_columns: list
            ... the columns based to do predictions of.
        target_columns: list
            ... the columns to predict.
        model_class: type
            ... a scikit-like model class to init with **kwargs


    """

    def __init__(
        self,
        feature_columns: List[str],
        target_columns: List[str],
        kernel: str = "rbf",
        alpha=1e-10,
        optimizer="fmin_l_bfgs_b",
        n_restarts_optimizer=0,
        normalize_y=False,
        copy_X_train=True,  # noqa
        random_state=None,
    ):
        super().__init__(
            feature_columns,
            target_columns,
            GaussianProcessRegressor,
            kernel,  # noqa
            alpha,  # noqa
            optimizer,  # noqa
            n_restarts_optimizer,  # noqa
            normalize_y,  # noqa
            copy_X_train,  # noqa
            random_state,  # noqa
        )

    def __str__(self):
        return f"Wrapper of model type {type(self.model)}"
