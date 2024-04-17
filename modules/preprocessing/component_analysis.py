import pandas as pd
from typing import List
from modules.preprocessing.scikit import SciKitPreProcessing
from sklearn.decomposition import FastICA, PCA, KernelPCA, FactorAnalysis
try:
    from sklearn.decomposition import MiniBatchNMF as NMF
except ImportError:
    from sklearn.decomposition import NMF


class CAPreProcessing(SciKitPreProcessing):
    def __init__(
        self,
        columns: List[str],
        suffix: str = "_ca",
        model_class=FactorAnalysis,
        **kwargs,
    ):
        """
        Components Analysis Preprocessing.

        Uses a SciKit like preprocessing with...

        - transform
        - inverse_transform

        This is specifically for component analyses:

        :math:`$\hat{x} = Ax, A ... demixing matrix$`

        NOTE: Here, all columns are considered part of the same feature!

        Output: M columns where the m'th columns represents the m'th component.
        they have "_<`suffix`>_m" in their name.

        :param columns: columns to decompose
        :param suffix: suffix for new columns names
        :param model_class: CA model class
        :param kwargs: CA model args as dictionary
        """
        self.cols = columns
        if "n_components" not in kwargs:
            kwargs["n_components"] = len(self.cols)
        self.kwargs = kwargs
        self.suffix = suffix
        super().__init__(columns, model_class, **kwargs)

    def transform(self, df: pd.DataFrame):
        n_comp = self.model.n_components
        col_name = (
            pd.Series(self.columns).str.replace("_[0-9]+$", "", regex=True).unique()[0]
        )
        cols = [f"{col_name + self.suffix}_{k}" for k in range(n_comp)]
        df[cols] = self.model.fit_transform(df[self.columns].values)
        return df


class ICAPreProcessing(CAPreProcessing):
    def __init__(self, columns: List, suffix: str = "_ic", **kwargs):
        super().__init__(columns, suffix, model_class=FastICA, **kwargs)


class PCAPreProcessing(CAPreProcessing):
    def __init__(self, columns: List, suffix: str = "_pc", **kwargs):
        super().__init__(columns, suffix, model_class=PCA, **kwargs)


class RBFPCAPreProcessing(CAPreProcessing):
    def __init__(self, columns: List, suffix: str = "_rbfpc", **kwargs):
        kwargs["n_jobs"] = -1
        kwargs["kernel"] = "rbf"
        if "gamma" not in kwargs:
            kwargs["gamma"] = 7 / 8 / len(columns)
        super().__init__(columns, suffix, model_class=KernelPCA, **kwargs)


class SigmoidPCAPreProcessing(CAPreProcessing):
    def __init__(self, columns: List, suffix: str = "_sigpc", **kwargs):
        kwargs["n_jobs"] = -1
        kwargs["kernel"] = "sigmoid"
        if "gamma" not in kwargs:
            kwargs["gamma"] = 7 / 8 / len(columns)
        super().__init__(columns, suffix, model_class=KernelPCA, **kwargs)


class PolyPCAPreProcessing(CAPreProcessing):
    def __init__(self, columns: List, suffix: str = "_polypc", **kwargs):
        kwargs["n_jobs"] = -1
        kwargs["kernel"] = "poly"
        if "gamma" not in kwargs:
            kwargs["gamma"] = 7 / 8 / len(columns)
        super().__init__(columns, suffix, model_class=KernelPCA, **kwargs)


class NMFPreProcessing(CAPreProcessing):
    def __init__(self, columns: List, suffix: str = "_nfm", **kwargs):
        if "l1_ratio" not in kwargs:
            kwargs["l1_ratio"] = 0.3
        super().__init__(columns, suffix, model_class=NMF, **kwargs)
