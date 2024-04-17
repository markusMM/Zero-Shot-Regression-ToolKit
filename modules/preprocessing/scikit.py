import pandas as pd
from typing import List
from modules.preprocessing.pre_processing import PreProcessing


class SciKitPreProcessing(PreProcessing):
    """
    Wrap any SciKit-Learn preprocessing.

    The handled class needs to be a model type with:
    - `fit` - taking a NumPy ndarray or Pandas DataFrame
    - `transform` - taking a NumPy ndarray or Pandas DataFrame
    - `fit_transform` - taking a NumPy ndarray or Pandas DataFrame

    The transform function automatically handles multi-outputs.

    NOTE: The new `columns` will be suffixed `_transformed` or `_transformed_<k>`
    respectively! (*<k>*, here, is a placeholder for the k'th multi-output!)

    :param columns: columns to be preprocessed (all separately)
    :param model_class: actual class of the SciKit model for preprocessing.
    :param drop_original: if deleting the original columns
    :param kwargs: additional params for the preprocessing model
    """
    def __init__(self, columns: List[str], model_class, drop_original=True, **kwargs):
        super().__init__(columns)
        self.model = model_class(**kwargs)
        self.drop = drop_original

    def fit(self, df: pd.DataFrame):
        self.model = self.model.fit(df[self.columns])
        return self.model

    def transform(self, df: pd.DataFrame):
        df_trans = []
        for c in self.columns:
            x = self.model.fit_transform(df[c])
            if x.shape[1] == 1:
                cols = c + "_transformed"
            else:
                cols = [f"{c}_transformed_{j}" for j in range(x.shape[1])]
            df_trans.append(pd.DataFrame(x, columns=cols))
        df = pd.concat([df] + df_trans, axis=1)
        if self.drop:
            df = df.drop(self.columns, axis=1)
        return df
