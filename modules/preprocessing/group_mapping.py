import pandas as pd
from modules.preprocessing import PreProcessing
from typing import Optional, Callable


class GroupFunPreProcessing(PreProcessing):
    """
    Group Function.

    Applies a function across the grouping over all given `columns`.

    NOTE: Here columns are the groupers themselves, and the function get run over the entrie data!
    It is considered to obtain a single output about that group! (E.g. `len`.)

    :param columns: columns to group by
    :param fun: function to extrac information from each group
    :param alias: alias for the new column
    """
    def __init__(
            self,
            columns,
            fun: Callable = len,
            alias: Optional[str] = None
    ):
        super().__init__(columns)
        self.alias = alias
        self.fun = fun

    def transform(self, df: pd.DataFrame) -> pd.DataFrame:
        tmp = df.groupby(self.columns).apply(self.fun)
        if self.alias is not None:
            tmp.name = self.alias
        else:
            tmp.name = '_'.join(self.columns) \
                       + f'_{self.fun.__name__.replace("<", "").replace(">", "")}'
        tmp = tmp.reset_index()
        return df.merge(tmp, on=self.columns)


class GroupedPreProc(PreProcessing):
    """
    Grouped Preprocessing.

    Wraps any type of preprocessing around a once per group processing and merges back
    the results.

    :param columns: columns to be grouped by
    :param pp_class: preprocssing class
    :param pp_params: preprocessing class init params
    """
    def __init__(
        self,
        columns: list,
        pp_class: type,
        pp_params: dict
    ):
        super().__init__(columns)
        self.pp = pp_class(**pp_params)

    def transform(self, df: pd.DataFrame):
        tmp = df.groupby(self.columns)[self.pp.columns].agg(
            lambda d: d.iloc[0]
        ).reset_index()
        tmp = self.pp.transform(tmp)
        df = df.merge(
            tmp, on=self.columns, how='left', suffixes=['', '_dropme']
        )
        return df.drop(list(filter(
            lambda c: '_dropme' in c,
            df.columns
        )), axis=1)
