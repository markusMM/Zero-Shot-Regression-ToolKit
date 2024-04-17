import re

import pandas as pd
import numpy as np
from typing import List, Dict
from modules.preprocessing import PreProcessing as Prep
from modules.preprocessing.aggregating import Aggregator
from modules.model import ModelWrapper
from skopt.space.space import Space
from skopt.space.space import Real
import joblib
import time
from modules.log import logger


class PipeLine:
    """
    The ML Pinpeline Class.

    - preprocess_data
    - fit
    - predict
    - recommend
    """

    def __init__(
        self,
        model_a: ModelWrapper,
        preprocessings_a: List[Prep],
        features_a: List[str],
        targets: List[str],
        model_b: ModelWrapper = None,
        preprocessings_b: List[Prep] = None,
        features_b: List[str] = None,
        model_b_crits: dict = None,
        reco_search_spaces: Dict[str, Space] = None,
        groupers: list = None,
        average_prediction: bool = True
    ):
        """
        ML-Pipe-Line object.

        It uses two model wrappers. One for main predictions and one more though as a backup.

        The backup `model_b` and all its related parameters are optional.

        At least the main model needs the following parameters:
            - preprocessings: A list of transformation steps for the dataframe.
            - features: A list of features to be considered within the dataframe.
            - targets: A list of all variables to be predicted. 
                (Need to exist in the data when training!)
                (Can essentially be NaN in small portions using semi-supervised learning!)

        It accepts defined search spaces in a dictionary with keys being the column
        and values being the actual search space.

        It is highly recommended to additionally label the search spaces, just in case!

        Args:
            model_a (ModelWrapper): the main model of type :class:`~modules.model.ModelWrapper`
            preprocessings_a (List[Prep]): a list of :class:`~modules.preprocessing.ProProcessing` for model_a
            features_a (List[str]): list of feature columns for the prediction using model_a
            targets (List[str]): the list of target variables to predict
            model_b (ModelWrapper, optional): same as model but for the alternate / backup model. 
                Defaults to None.
            preprocessings_b (List[Prep], optional): same as preprocessing but for alternate model. 
                Defaults to None.
            features_b (List[str], optional): list of feature columns for the prediction using model_b. 
                Defaults to None.
            model_b_crits (dict, optional): a dictionary of criteria whether to use alternate model. 
                Defaults to None.
            reco_search_spaces (Dict[str, Space], optional): a dictionary of type str and :class:`~skopt.Space`. 
                In case of using a recommendation searcher. Defaults to None.
            groupers (list, optional): groupers for the data to aggregate to, if average prediction. 
                Defaults to None.
            average_prediction (bool, optional): whether to average the final prediction or not. 
                Defaults to True.
        """
        if model_b_crits is None:
            model_b_crits = {
                'creative_id': 0
            }
        if reco_search_spaces is None:
            reco_search_spaces = {}
        self.model_a = model_a
        self.model_b = model_b
        self.preprocessings_a = preprocessings_a
        self.preprocessings_b = preprocessings_b
        self.features_a = features_a
        self.features_b = features_b
        self.targets = targets
        self.re_spaces = reco_search_spaces
        self.model_b_crits = model_b_crits
        self.average_prediction = average_prediction
        self.pp_agg_idx = None
        self.pp_agg_funs = None
        agg_idx = None
        if len(self.preprocessings_a) > 0:
            for p_index, pp in enumerate(self.preprocessings_a):
                if isinstance(pp, Aggregator):
                    agg_idx = p_index
                    break
            if agg_idx is not None and groupers is None:
                groupers = getattr(
                    self.preprocessings_a[agg_idx],
                    'indices',
                    []
                )
        self.pp_agg_idx = agg_idx
        if groupers is None:
            groupers = []
        self.groupers = groupers

    def load(self, model_path):
        """
        Joblib load self (`PipeLine`) from `model_path`.

        This method currently uses joblib to load the entire pipeline
        fom a file called `model.tar.gz`, if no file is given.
        WARNING: The original object does not change! It returns a new instance!
        """
        if len(re.findall('\..+$', model_path)) <= 0:
            model_path = model_path + '/model.tar.gz'
        return joblib.load(model_path)

    def save(self, model_path):
        """
        Pickle self (`PipeLine`) into `model_path`.

        This method currently uses joblib to dump the entire pipeline
        into a file called `model.tar.gz`, if no file is given.
        """
        if len(re.findall('\..+$', model_path)) <= 0:
            model_path = model_path + '/model.tar.gz'
        joblib.dump(self, model_path, compress=2)

    def check_tabular(self, df):
        tab_mod = np.zeros((len(df))).astype(bool)
        if self.model_b is None:
            return tab_mod
        for c in self.model_b_crits:
            tab_mod |= (df[c].values == self.model_b_crits[c])
        return tab_mod

    def get_avg_cols(self):
        """
        Get average columns.

        In case these columns are not set already, they are sought from preprocessings.
        It refers to the 'c_agg' attribute of :class:`~modules.preprocessing.Aggregator`

        :return: average columns
        """
        avg_cols = getattr(self, 'avg_cols')
        if self.pp_agg_funs is None and avg_cols is None:
            self.pp_agg_funs = getattr(
                self.preprocessings_a[self.pp_agg_idx],
                'c_agg',
                {}
            )
            self.avg_cols = list(filter(
                lambda c:
                ('mean' in self.pp_agg_funs[c].__name__ or
                 'average' in self.pp_agg_funs[c].__name__ or
                 'avg' in self.pp_agg_funs[c].__name__) and c in self.targets,
                self.pp_agg_funs.keys()
            ))
        if len(avg_cols) <= 0:
            self.average_prediction = False
        else:
            self.average_prediction = True
        self.avg_cols = avg_cols
        return avg_cols

    @staticmethod
    def process_data__(df, preprocessing):
        for p in preprocessing:
            logger.info(f'Processing {p}... ')
            t = time.time_ns()
            df = p.transform(df)
            dt = time.time_ns() - t
            logger.info(f'done after {dt * 1e-9}')
        return df

    @staticmethod
    def exp_log_prediction(pred: pd.DataFrame) -> pd.DataFrame:
        """
        Both check and transform logarithmic outputs.

        For resilience reasons, non-finite results get parsed.
        This parsing tries to approach realistic values by altering the log-values.
        If this fails, after 20 tries, those log-values get simply multiplied by 10k!

        :param pred: predictions of `self.model`
        :return: predictions with all "log_" values being transformed exponentially
        """
        log_cols = pred.columns[
            pred.columns.str.contains('^log_', regex=True)
        ]

        if len(log_cols) > 0:
            for c in log_cols:
                log_vals = pred[c].values
                exp_vals = np.expm1(pred[c].values)
                n = 1

                # trying to alter odd log values
                while sum(~np.isfinite(exp_vals)) and n < 20:
                    log_vals[
                        ~np.isfinite(exp_vals) & (log_vals > 1)
                    ] = log_vals[~np.isfinite(exp_vals) & (log_vals > 1)] - .10
                    log_vals[
                        ~np.isfinite(exp_vals) & (log_vals < 1)
                    ] = log_vals[~np.isfinite(exp_vals) & (log_vals < 1)] + .10
                    exp_vals[~np.isfinite(exp_vals)] = np.expm1(
                        log_vals[~np.isfinite(exp_vals)]
                    )
                    n += 1

                # for all fails from altering, multiply log values by 10k
                exp_vals[~np.isfinite(exp_vals)] = np.abs(log_vals[~np.isfinite(exp_vals)]) * 10_000  # noqa
                pred[c.replace('log_', '')] = exp_vals

                # delete old log-values,
                # because they are not part of the output schema.
                del pred[c]
        return pred

    def preprocess_data(self, df: pd.DataFrame, tab_mod=None):
        """
        Preprocess data, according to different preprocessings.

        For tabular model, it just uses tabular preprocessing, if set.

        :param df: data
        :param tab_mod: Boolean array for each row, to be rather tabular or not.
        :return: pre-processed data
        """
        tab_preproc = getattr(self, 'tabular_preprocessing', None)
        if tab_mod is None:
            tab_mod = self.check_tabular(df)
        if sum(~tab_mod) > 0 or tab_preproc is None:
            df = self.process_data__(df, self.preprocessings_a)
            for c in set(self.features_a).difference(df.columns):
                df[c] = 0
        else:
            df = self.process_data__(df, tab_preproc)
        return df.fillna(0)

    def predict(self, df: pd.DataFrame):
        """
        Prediction.

        > Process:
            1. Ensure the NN models are in evaluation mode.
            2. Extract target columns for prediction.
            3. Determine if the data is tabular or not and handle grouping if applicable.
            4. Preprocess the data, checking for potential data aggregation.
            5. Initialize a DataFrame to store the predictions.
            6. Make predictions for tabular and non-tabular data separately.
            7. Exponentiate log-predictions, if any.
            8. Handle averaging predictions if necessary.
            9. Remove target columns from original df.
            10. Concatenate original with the prediction.

        > For aggregated models

        Here, we try to find all kinds of averaging during preprocessing.
        When `avg_cols` was not set from outside,
        we look for the name of the functions used to aggregate.
        This is done to extrapolate the final prediction back into a sum,
        using the number of instances obtained by the `groupers`.
        It refers to the 'c_agg' attribute of :class:`~modules.preprocessing.Aggregator`

        **NOTE**: This is one reason for the importance of declaring **`groupers`**!

        :param df: data
        :return: df | prediction
        """
        if hasattr(self.model_a.model, 'eval'):  # e.g. a PyTorch module
            self.model_a.model = self.model_a.model.eval()
        if hasattr(self.model_a.model, 'eval'):
            self.model_b.model = self.model_b.model.eval()
        ycols = self.targets
        if isinstance(ycols, str):
            ycols = [ycols]
        n_instances = len(df)  # this value is to check for aggregation in pp later
        tab_mod = self.check_tabular(df)
        groupers = getattr(self, 'groupers', [])
        self.groupers = list(filter(lambda c: c in df.columns, groupers))
        if len(groupers) > 0:
            inst_cnt = df.groupby(groupers).count().iloc[:, 0]
        else:
            inst_cnt = len(df)

        df = self.preprocess_data(df, tab_mod)
        if len(df) < n_instances:  # in case of an aggregation in pre-processing
            tab_mod = self.check_tabular(df)

        pred = pd.DataFrame(
            [[0.0] * len(ycols)] * df.shape[0],
            columns=ycols,
            index=df.index
        )
        if sum(tab_mod) > 0:
            if self.model_b is None:
                self.model_b = self.model_a
            x = df.loc[tab_mod]
            y = self.model_b.predict(x)
            y = y[ycols]
            pred.loc[tab_mod.tolist(), ycols] = y.values
        if sum(~tab_mod) > 0:
            x = df.loc[~tab_mod]
            y = self.model_a.predict(x)
            y = y[ycols]
            pred.loc[(~tab_mod).tolist(), ycols] = y.values

        pred = self.exp_log_prediction(pred)
        ycols = pred.columns

        if len(pred) < n_instances:
            for c in self.get_avg_cols():
                pred[c] *= inst_cnt

        for t in ycols:
            if t in df.columns:
                df = df.drop(t, axis=1)
        return pd.concat([
            df,
            pd.DataFrame(pred, columns=ycols)
        ], axis=1)

    def predict_metrics(self, df: pd.DataFrame):
        return self.predict(df)

    def fit_model(self, df: pd.DataFrame):
        return self.fit(df)

    def fit(self, df: pd.DataFrame):
        """
        Fit model(s).

        This model is supposed to fit both models.
        But the tabular only if it was set.

        This method just inherently calls the models' fit function.

        :param df: data to fit the models on
        :return: self, to preserve the SciKit spirit / analogy
        """
        df = self.preprocess_data(df)
        self.model_a = self.model_a.fit(df)
        if self.model_b is not None and hasattr(self.model_b, 'fit'):
            self.model_b = self.model_b.fit(df)
        return self

    def parse_search_spaces(self, re_cols):
        space = []
        for c in re_cols:
            space.append(self.re_spaces.get(c, Real(-200, 200, name=c)))
        return space

    def argmin_rec(self, z, df, target_col, argmax: bool = True):
        """
        Argmin function, which returns a form of the KPI to be minimized.

        So, is if argmax is true, it returns 1 / KPI instead!
        :param z: a dictionary with the sample values to be changed / recommended
        :param df: the original data
        :param target_col: target KPI to be optimized
        :param argmax: boolean of whether to maximize the KPI or not
        :return:
        """
        reco = np.array(list(z.values())*len(df))
        if len(reco.shape) > 2:
            reco = reco.squeeze()
        if len(reco.shape) < 2:
            reco = reco[:, None]
        df[list(z.keys())] = reco  # is dim(N, M)
        df = self.predict(df)
        if argmax:
            return 1 / np.nanmean(df[target_col].values)
        else:
            return np.nanmean(df[target_col].values)

    def recommend(
        self, df: pd.DataFrame,
        re_cols: List[str],
        target_col: str,
        argmax: bool = True
    ):
        """
        Recommend better adjustments from sampling.

        This function samples random numbers from a sampling space for each specified
        column to optimize.
        It always just generates exactly 25x80 samples.
        It finally adds the expectation values of the best 25 to the dataframe.
        Then the columns get returned.

        The expectation value is taken from guassian distance, for p=0.05, from the
        optimum times the gaussian sample distribution as approximate posterior.

        :param df: data containing all features to forecast
        :param re_cols: columns to change
        :param target_col: column to optimize wrt
        :param argmax: whether it shall be maximized
        :return: dataframe with just the optimized columns.
        """
        if isinstance(re_cols, str):
            re_cols = [re_cols]
        smpl_states = []
        smpl_metric = []
        space = self.parse_search_spaces(re_cols)

        for _ in range(80):
            states = []
            metric = []
            for _ in range(25):
                z = {s.name: s.rvs(1) for s in space}
                states.append(np.array(list(z.values())))
                metric.append(self.argmin_rec(z, df, target_col, argmax))
            metric = np.array(metric).squeeze()
            # appending the state and metric
            smpl_metric.append(metric.min())
            smpl_states.append(states[metric.argmin()])

        smpl_metric = np.array(smpl_metric).squeeze()
        smpl_states = np.array(smpl_states).squeeze()
        smpl_likelihood = -((smpl_metric.min() - smpl_metric) ** 2) / 0.05
        smpl_prior = -((smpl_states.mean(0) - smpl_states) ** 2) / smpl_states.var()
        smpl_expect = np.exp(smpl_likelihood) * np.exp(smpl_prior) * smpl_states
        smpl_expect /= (np.exp(smpl_likelihood) * np.exp(smpl_prior)).sum()

        df[re_cols] = smpl_expect.mean(0)

        return df[re_cols]
