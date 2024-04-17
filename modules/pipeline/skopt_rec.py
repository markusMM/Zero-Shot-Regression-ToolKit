import numpy as np
import pandas as pd
from functools import partial
from typing import List, Dict
from modules.preprocessing import PreProcessing as Prep
from modules.model import ModelWrapper
from modules.pipeline.pipe_line import PipeLine
from skopt.space.space import Space
from skopt import gp_minimize, space


class SkOptRecPipe(PipeLine):
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
        SciKit-Optimize Recommender

        This is the same module as :class:`~modules.pipeline.pipe_line.PipeLine`,
        but it uses SciKit-Optimize to minimize or maximize a specific target variable!

        Here, there serach spaces are directly usable by the `skopt.gp_minimize` function
        embedded into this module.

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
        super().__init__(
            model_a,
            preprocessings_a,
            features_a,
            targets,
            model_b,
            preprocessings_b,
            features_b,
            model_b_crits,
            reco_search_spaces,
            groupers,
            average_prediction
        )

    def argmin_rec(
            self,
            sample,
            df,
            re_cols,
            target_col,
            argmax: bool = True
    ):
        """
        Argmin function for SK-Optimize GP minimization.

        This function ensures that the objective is minimized wrt model prediction.

        So, if `maximize` is true, it set it to 1 / mean of the target.
        Additionally, 0.7 * sample corr is added to ensure some sparsity.

        :param sample: single sample for all the values to be optimizes
        :param df: data in which these values get replaced
        :param re_cols: recommendation column names
        :param target_col: target KPI column
        :param argmax: whther to maximize target or not
        :return: either mean or 1 / mean of final target
         time (0.7 + 0.3 * sample corr)
        """
        sample = pd.DataFrame(sample, columns=re_cols)
        sample = sample.fillna(df)
        df[sample.columns] = sample.values
        df = self.predict(df)
        c = sample.values
        c = c.T @ c / (c**2).sum(0)
        if argmax:
            return (.7 + .3 * np.nanmean(c)) / np.nanmean(df[target_col])
        else:
            return (.7 + .3 * np.nanmean(c)) * np.nanmean(df[target_col])

    @staticmethod
    def get_search_bounds(df, re_cols, target_col):
        """
        Improvised defaulting of search boundaries.

        This function is used in case, we do not have a search space given for a value.
        It mostly just uses random values as mean and deviation.
        But if the target KPI is given and has non-empty entries,
        the mean and standard deviation of the KPI is used.

        :param df: data (containing all necessary values for prediction)
        :param re_cols: recommendation columns
        :param target_col: the name of the KPI to me optimized
        """
        m = pd.DataFrame(
            [[np.random.randn()*100]*len(re_cols)],
            columns=re_cols
        )
        v = pd.DataFrame(
            [[np.random.randn()*60]*len(re_cols)],
            columns=re_cols
        )
        if target_col in df:
            if sum(~df[target_col].isna()) > 0:
                m = pd.DataFrame(
                    [np.abs(np.nanmean(df[target_col], axis=0))],
                    columns=re_cols
                )
                v = pd.DataFrame(
                    [np.abs(np.nanstd(df[target_col], axis=0))],
                    columns=re_cols
                )
        return m, v

    def recommend(
        self,
        df: pd.DataFrame,
        re_cols: List[str],
        target_col: str,
        argmax: bool = True
    ) -> pd.DataFrame:
        """
        Optimize variables according to KPI using GP minimization.

        Uses SKOpt GP minimization from SciKit-Opt.
        :func:`~skopt.gp_minimize`

        Default params are:

            n_jobs=-1
            n_points=20
            n_calls=15

        :param df: data
        :param re_cols: reco columns
        :param target_col: target column
        :param argmax: whether to maximize or not
        :return: data with new optimized columns
        """

        # get mean and deviation for default search space placeholders
        m, v = self.get_search_bounds(df, re_cols, target_col)

        # each search space is obtained with a getter,
        # using a real value space as default
        search = gp_minimize(
            partial(
                self.argmin_rec,
                df=df,
                re_cols=re_cols,
                target_col=target_col,
                argmax=argmax,
            ),
            [
                self.re_spaces.get(
                    re_col,
                    space.Real(
                        (m[re_col] - 1.5*(v[re_col] + m[re_col])).values,
                        (m[re_col] + 1.5*(v[re_col] + m[re_col])).values,
                        name=re_col
                    )
                )
                for re_col in re_cols
            ],
            n_jobs=-1,
            n_points=20,
            n_calls=15
        )
        df[re_cols] = search["x"]
        return df[re_cols]
