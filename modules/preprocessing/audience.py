import pandas as pd
from modules.preprocessing.util import decifer_age_targets
from modules.preprocessing.pre_processing import PreProcessing


class AgeHistPreProcessing(PreProcessing):

    def __init__(self, columns=None):
        """
        Age Histograms.

        Makes use of :meth:`~modules/preprocessing/util/decifer_age_targets`.

        It collects all age groups, sums them together (as ones) and norms them.
        So, for each instance, there is a histogram based on the age targets.
        (Usually 1 instance is a creative of a line item!)

        :param columns: not necessary, just inherited
        """
        super().__init__(columns=columns)

    def transform(self, df: pd.DataFrame):
        return decifer_age_targets(df, False)
