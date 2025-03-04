import distutils.dir_util
import joblib
import glob
from typing import Any, Callable, Literal, Optional, Union
from sklearn.metrics import r2_score as r2
from sklearn.kernel_ridge import KernelRidge
import pandas as pd
import numpy as np
from torch.utils.data import DataLoader
from torch import Tensor
from sklearn.preprocessing import MinMaxScaler

from modules.preprocessing import CosPeriod
from modules.log import logger


class ModelWrapper:
    """ Model Wrapper

    Does everything a model should do with...
     - fit
        ... fits the underlying model and returns this object.
            in: DataFrame with `feature_columns` and `target_columns`
            out: self (no copy for now)
     - predict
        ... predicts `target_columns` given a DataFrame of containing
        `feature_columns`
     - predict proba
        ... returns the underlying model's probabilities/confidences
        for some value prediction.
            in: DataFrame with `feature_columns`
            out: DataFrame with Probabilities for each of `target_columns`

    params:
        feature_columns (List[str]): The columns based to do predictions of.
        target_columns (List[str]): The columns to predict.
        score_fun (Callable[[Any, Any], float]): Native scoring function of the model.
            Defaults to `r2`, the R2 norm from `sklearn.metrics`.
        model (object): Any estimator like model with a fit and a predict function.
            Defaults to `KernelRidge` with RBF kernel.
    """
    def __init__(
            self,
            feature_columns: list,
            target_columns: list,
            score_fun: Callable[
                [Any, Any], 
                Union[float, np.ndarray, Tensor]
            ] = r2,  # type: ignore
            model: object = KernelRidge(.98, kernel="rbf")
    ) -> None:
        self.std = 0.05
        self.features = feature_columns
        self.targets = target_columns
        self.score_fun = score_fun
        self.model = model

    def save_obj(
        self,
        obj=None,
        path: str = "./models/model.pkl"
    ):
        if obj is None:
            obj = self
        existing = glob.glob(path)
        if len(existing):
            path = path.rsplit(".", 1)  # type: ignore
            path = path[0] + f"_{len(existing) + 1}." + path[1]
        distutils.dir_util.mkpath("/".join(path.split("/")[:-1]))
        joblib.dump(obj, path)

    def save_model(self, path="./model_checkpoints/model.pkl") -> None:
        """Abstract method simply pickling the submodel."""
        joblib.dump(self.model, path)

    def save_params(self, path: str) -> None:
        """Abstract method storing model artifacts."""
        params = {}
        if hasattr(self.model, "get_params"):
            params["coefs"] = self.model.get_params()  # type: ignore
        elif hasattr(self.model, "state_dict"):
            params["coefs"] = self.model.state_dict()  # type: ignore
        else:
            logger.warn(
                f"Cannot parse coefs for model class {self.model.__class__}!"
            )
        
        params["std"] = self.std
        params["feature_columns"] = self.features
        params["target_columns"] = self.targets
        params["score_fun"] = self.score_fun

        joblib.dump(params, path)

    def confidence(self, df: pd.DataFrame) -> pd.DataFrame:
        """Data confidences based on features."""
        return (
            self.std * np.sqrt(
                1 + (self.mu_x - df[self.features]) ** 2 / self.va_x
            )
        ).mean(1)

    def _check_features_(self, df: pd.DataFrame) -> pd.DataFrame:
        diff = list(set(self.features).difference(df.columns))
        if len(diff) > 0:
            df[diff] = 0
        return df

    def predict(self, df: pd.DataFrame) -> pd.DataFrame:
        """Predict target metrics based on features."""
        y = self.model.predict(df[self.features])  # type: ignore
        return pd.DataFrame(y, columns=self.targets)

    def predict_variance_test(self, df) -> dict[str, Any]:
        minmax = MinMaxScaler()
        def get_minmax_var(x): return (minmax.fit_transform(x)).var(axis=0)
        vx = get_minmax_var(df[self.features].values)
        vy = get_minmax_var(df[self.targets].values)
        vp = get_minmax_var(self.predict(df).values)
        dv_yp = pd.Series(
            np.abs(vy - vp) / vy,
            index=self.targets
        )
        dv_xp = pd.DataFrame(
            np.abs(vx[:, None] - vp[None]) / vx[:, None],
            index=self.features,
            columns=self.targets
        )
        return {
            'feature2prediction_variance': dv_xp,
            'target2prediction_variance': dv_yp
        }

    def fit(self, df: pd.DataFrame) -> "ModelWrapper":
        """
        Abstract class to fit underlying model.

        NOTE: This is an abstract function, just updating
              mean and variance of the features!

        params:
            df (Dataframe): Data to fit the model on.
        """
        self.model = self.model.fit(df[self.features], df[self.targets])  # type: ignore
        self.mu_x = np.nanmean(df[self.features].values, axis=0)
        self.va_x = np.nanvar(df[self.features].values, axis=0)
        return self

    def scores(self, df, ):
        p = self.predict(df)
        y = df[self.targets]
        scores = {}
        for target in self.targets:
            scores[target] = self.score_fun(
                p[target].values,
                y[target].values
            )
        return scores

    def score(self, df):
        p = self.predict(df)
        y = df[self.features]
        return self.score_fun(
            p.values,
            y.values
        )

    def __str__(self) -> Literal['Blank Model Wrapper']:
        return "Blank Model Wrapper"


class GroupedModelWrapper:
    """ Model Wrapper

    Does everything a model should do with...
     - fit
        ... fits the underlying model and returns this object.
            in: DataFrame with `feature_columns` and `target_columns`
            out: self (no copy for now)
     - predict
        ... predicts `target_columns` given a DataFrame of containing
        `feature_columns`
     - predict proba
        ... returns the underlying model's probabilities/confidences
        for some value prediction.
            in: DataFrame with `feature_columns`
            out: DataFrame with Probabilities for each of `target_columns`

    params:
        feature_columns: list
            ... the columns based to do predictions of.
        target_columns: list
            ... the columns to predict.
    """

    def __init__(
            self,
            feature_columns: list,
            target_columns: list,
            grouper: str,
            score_fun: Callable = r2
    ) -> None:
        self.std = 0.05
        self.features = feature_columns
        self.targets = target_columns
        self.grouper = grouper
        self.score_fun = score_fun
        self.model = {}

    def save_obj(
        self,
        obj=None,
        path: str = "./models/model.pkl"
    ) -> None:
        if obj is None:
            obj = self
        existing = glob.glob(path)
        if len(existing):
            path = path.rsplit(".", 1)  # type: ignore
            path = path[0] + f"_{len(existing) + 1}." + path[1]
        distutils.dir_util.mkpath("/".join(path.split("/")[:-1]))
        joblib.dump(obj, path)

    def save_model(self, path="./model_checkpoints/model.pkl"):
        self.save_obj(self.model, path)

    def save_params(self, path: str):
        """A method for storing model artifacts."""
        pass

    def confidence(self, df: pd.DataFrame) -> pd.DataFrame:
        """Data confidences based on features."""
        return np.array([(
            self.std * np.sqrt(
                1 + (self.mu_x - d[1].values) ** 2 / self.va_x  # type: ignore
            )
        ).mean(1) for d in df.groupby(self.grouper)[self.features]])

    def _check_features_(self, df: pd.DataFrame) -> pd.DataFrame:
        diff = list(set(self.features).difference(df.columns))
        if len(diff) > 0:
            df[diff] = 0
        return df

    def predict(self, df: pd.DataFrame) -> pd.DataFrame:
        """Predict target metrics based on features."""
        df = self._check_features_(df)
        mocks = []
        imp_col = 'imps'
        click_col = 'clicks'
        conv_cols = []
        for t in self.targets:
            if 'imp' in t and 'unique' not in t:
                if 'conv' in t:
                    mocks.append(np.random.randint(4, 20, (len(df), 1)))
                    conv_cols.append(t)
                else:
                    mocks.append(np.random.randint(1000, 2000, (len(df), 1)))
                    imp_col = t
            elif 'click' in t:
                mocks.append(np.random.randint(0, 20, (len(df), 1)))
                if 'conv' in t:
                    mocks.append(np.random.randint(6, 50, (len(df), 1)))
                    conv_cols.append(t)
                else:
                    mocks.append(np.random.randint(0, 20, (len(df), 1)))
                    click_col = t
            elif 'conv' in t and 'click' not in t and 'imp' not in t:
                mocks.append(np.random.randint(4, 50, (len(df), 1)))
                conv_cols.append(t)
            else:
                mocks.append(np.random.randint(4, 50, (len(df), 1)))
        mocks = np.concatenate(mocks, axis=1)
        mocks = pd.DataFrame(
            mocks[:, :len(self.targets)],
            columns=self.targets,
            index=df.index,
        )
        if click_col in mocks and imp_col in mocks:
            mocks[click_col] = mocks[click_col] & mocks[imp_col]
        for c in conv_cols:
            if c in mocks:
                if 'imp' in c and imp_col in mocks:
                    mocks[c] = mocks[c] & mocks[imp_col]
                elif click_col in mocks:
                    mocks[c] = mocks[c] & mocks[click_col]
        return mocks

    def fit(self, df: pd.DataFrame) -> "GroupedModelWrapper":
        """Fit underlying model."""
        self.mu_x = [
            np.nanmean(d[1].values, axis=0)
            for d in df.groupby(self.grouper)[self.features]
        ]
        self.va_x = [
            np.nanvar(d[1].values, axis=0)
            for d in df.groupby(self.grouper)[self.features]
        ]
        return self

    def scores(self, df) -> dict[Any, Any]:
        p = self.predict(df)
        y = df[self.features]
        scores = {}
        for feature in self.features:
            scores[feature] = self.score_fun(
                p[feature].values,
                y[feature].values
            )
        return scores

    def score(self, df) -> Any:
        p = self.predict(df)
        y = df[self.features]
        return self.score_fun(
            p.values,
            y.values
        )

    def __str__(self):
        return "Blank Model Wrapper"


class FactorizedModel(ModelWrapper):
    """
    Factorized Model Wrapper.

    Args:
            feature_columns (list): _description_
            target_columns (list): _description_
            score_fun (Callable, optional): _description_. Defaults to r2.
            model_class (Optional[type], optional): Class of the underlying model. 
                If None is given, the default model is a 
            model_params (dict, optional): Parameters for the underlying model. 
                Defaults to {}.
            model_xy_split (bool, optional): Whether to split the data in X and Y for 
                the underlying model's fit. Defaults to True.
            factor_models (_type_, optional): A dictionary of factorization functions.
                The keys, here, are the variables to run the factorization on. 
                Defaults to { 'hour': CosPeriod( [f'hour_{j}' for j in range(24)], 16 ) }.
                Note: _CosPeriod_, here, is the predefined function 
                      `modules.preprocessing.CosPeriod`!
    """

    def __init__(
        self,
        feature_columns: list,
        target_columns: list,
        score_fun: Callable = r2,
        model_class: Optional[type] = None,
        model_params: dict = {},
        model_xy_split: bool = True,
        factor_models: dict = {
            'hour': CosPeriod(
                [f'hour_{j}' for j in range(24)],
                16
            )
        }
    ) -> None:
        super().__init__(feature_columns, target_columns, score_fun)

        self.factor_models = factor_models
        self.model = model_class(**model_params)  # type: ignore
        self.xy_split = model_xy_split

    def fit(
        self, 
        df: Union[pd.DataFrame, DataLoader]
    ) -> "FactorizedModel":
        """
        Fit the model!

        NOTE: The model needs a function called "fit" for this to work!

        Args:
            df (_type_): _description_

        Returns:
            FactorizedModel: _description_
        """
        if not self.xy_split:
            self.model = self.model.fit(df)
        else:
            self.model = self.model.fit(
                df[self.features].values,  # type: ignore
                df[self.targets].values  # type: ignore
            )
        return self

    def predict(self, df) -> pd.DataFrame:
        df = self._check_features_(df)
        fac = np.ones(len(df))
        for fm in self.factor_models:
            fm = self.factor_models[fm]
            cols_ = df.columns
            # abstract factor model handler
            # whether it is a preprocessing or prediction model
            if hasattr(fm, 'transform'):
                df = getattr(fm, 'transform')(df)
                f = df[list(set(df.columns).difference(cols_))].sum(axis=1).values
            else:
                f = getattr(
                    fm, 'predict',
                    getattr(
                        fm, 'forward',
                    )
                )(df).sum(axis=1)
            # factorize, if at least anything have been given
            f = np.array(f.tolist())
            f[f <= 0] = 1
            fac *= f

        pred = self.model.predict(df[self.features])
        if not isinstance(pred, pd.DataFrame):
            pred = pd.DataFrame(
                pred,
                columns=self.features
            )
        return pred * fac[:, None]


class FeatureDefaultWrapper(ModelWrapper):

    def __init__(
            self,
            feature_columns,
            target_columns,
            model,
            default_features: dict = {
                'language_': True,
                'browser_targets_': True,
                'os_targets_': True,
                'ad_type_': True
            }
    ):
        super().__init__(feature_columns=feature_columns, target_columns=target_columns)
        self.model = model
        self.defaults = default_features

    def set_defaults(self, df):
        for d in self.defaults:
            df[list(filter(
                lambda c: d in c,
                df.columns
            ))] = self.defaults[d]
        return df

    def predict(self, df: pd.DataFrame) -> pd.DataFrame:
        df = self.set_defaults(df)
        if isinstance(self.model, ModelWrapper):
            return self.model.predict(df)
        else:
            return self.model.predict(df[self.features].values)

    def fit(self, df: pd.DataFrame):
        df = self.set_defaults(df)
        if isinstance(self.model, ModelWrapper):
            return self.model.fit(df)
        else:
            return self.model.fit(
                df[self.features].values,
                df[self.targets].values
            )
