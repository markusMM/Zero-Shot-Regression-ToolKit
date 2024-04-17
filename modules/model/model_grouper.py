import numpy as np

from modules.log import logger
from modules.model.model_wrapper import GroupedModelWrapper
import pandas as pd
from typing import Callable
from sklearn.metrics import r2_score as r2


class ModelGrouper(GroupedModelWrapper):

    def __init__(
            self,
            features,
            targets,
            grouper,
            model_class,
            model_args,
            group=None
    ):
        """
        Group the models.

        This is a model grouper. It is meant to work as a normal model.
        But it is a model of models based on the `grouper` and the resulting `group`
        from the data.

        :param list features:
        List of features to be considered in the incomming dataframes.
        :param list targets: List of targets from incomming dataframes,
        the models have to be fit on.
        :param str grouper: Name of the columns, the data has to be grouped on.
        This defines the final number of models inside this grouper.
        :param None model_class: Estimator class to be used for each model.
        It is recommended to use a `ModelWrapper` module for this.
        :param dict model_args: The initialization arguments for the model class.
        :param list group: optional,
        A predefined group for the final grouping of the models.
        """
        ma_keys = list(set(model_args.keys()).difference(
            ['features', 'targets']
        ))
        model_args = {ma: model_args[ma] for ma in ma_keys}
        super().__init__(features, targets, grouper)
        self.grouper = grouper
        self.model_class = model_class
        self.model_args = model_args
        self.group = group
        self.model = {}

    def fit(self, df: pd.DataFrame):
        """
        Fit each and define model based on `groups`.

        :param df: Input data.
        NOTE: It needs to contain both features and targets.
        :return self: This object. NOTE: This is not an object copy!
        """
        if self.group is None:
            self.group = df[self.grouper].unique()
        self.group = np.sort(self.group)
        df = df.sort_values(self.grouper)
        super().fit(df)

        for g in self.group:
            self.model[g] = self.model_class(
                self.features, self.targets, **self.model_args
            )
            try:
                self.model[g] = self.model[g].fit(df)
            except Exception as e:  # noqa
                logger.warn(e)
                logger.warn(f'Cannot train {self.model_class} on {self.grouper} {g}.')

        return self

    def predict(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Predict models output.

        Different from the original predict function for a non-grouped model,
        here there is a prediction for each individual instance of the group.
        the final prediction will then be concatenated.

        WARNING: The order of the output may vary from the input in the indexing.
        (The index is kept!)

        :param df: pd.DataFrame,
        The dataframe with all the necessary input features to perform a prediction.
        :return prediction: pd.DataFrame,
        A dataframe with all targets from `self.targets` for each instance of the input.
        """
        prediction = []
        for g, df in df.groupby(self.grouper):
            prediction.append(
                pd.DataFrame(
                    self.model[g].predict(df),
                    index=df.index,
                    columns=self.targets
                )
            )
        return pd.concat(prediction, axis=0)

    def scores(self, df: pd.DataFrame, fun: Callable = r2) -> pd.DataFrame:
        """
        Evaluate the models upon their targets.

        This function is meant to evaluate all inmating models upon all targets.
        This is done with the evaluation function 'fun'.

        NOTE: The input dataframe needs both, all features and all targets to be set.

        :param df: The input dataframe with all features and targets for evaluation.
        :param fun: The evaluation function. Default: R²
        :return scores: A targets x groups dataframe containing the respective scores.
        """
        assert sum(map(lambda t: t not in df, self.targets)) <= 0
        score = []
        for g, df in df.groupby(self.grouper):
            scr = pd.DataFrame(columns=self.targets, index=[g])
            prd = self.model[g].predict(df)
            for t in self.targets:
                scr[t] = fun(prd[t].values, df[t].values)
            score.append(
                scr
            )
        return pd.concat(score, axis=0)

    def score(self, df: pd.DataFrame, fun: Callable = r2):
        """
        Wrap up the scores.

        This is by far not the most accurate solution, but it averages all scores from
        the `scores` function.

        The idea is to get one score for all models in the group to evaluate the entire
        array within the runs as if it would be a single multiple regression.

        :param pd.DataFrame df: Input data.
        NOTE: It has to contain both, all features and all targets respectively.
        :param Callable fun: optional, Scoring function for the individual cases.
        :return float score: The final average of all targets x groups scores.
        """
        return np.nanmean(self.scores(df, fun).values)
