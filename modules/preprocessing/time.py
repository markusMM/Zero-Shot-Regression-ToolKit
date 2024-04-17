from typing import List, Any, Optional, Dict, Union, Iterable

import numpy as np
import pandas as pd
import multiprocessing
from functools import partial
from modules.log import logger
from modules.preprocessing.factorize import Factorize
from modules.preprocessing.pre_processing import PreProcessing
from modules.preprocessing.util import kgauss
from modules.common import NCPU


class WeekdayPreProcessing(PreProcessing):
    """
    Extract weekday information from datetime columns.

    Used to extract weekday information from datetime columns in the data and create
    'weekday' columns to store the names.

    Parameters:
    :param columns: A list of column names containing datetime data from which weekdays will be extracted.
                    (List[str])

    Methods:
    :return: The DataFrame with weekday information extracted and stored in new columns.
             (pd.DataFrame)

    Example:

    >>> processor = WeekdayPreProcessing(columns=['date_column'])
    >>> df_processed = processor.transform(df)
    >>> print(df_processed['weekday'].head(3).values)
    ... array(['Monday', 'Monday', 'Tuesday'])
    """
    def __init__(self, columns: List):
        super().__init__(columns)

    def transform(self, df: pd.DataFrame):
        """
        Transform the DataFrame by extracting weekday information from datetime columns.

        Converts datetime columns to datetime objects, extracts weekday names into a new
        'weekday' column

        :param df: The DataFrame containing datetime columns to extract weekday information from.
                   (pd.DataFrame)
        :return: The transformed DataFrame with weekday information extracted and stored in new columns.
                 (pd.DataFrame)
        """
        for c in self.columns:
            col = '_'.join(c.split("_")[:-1]) + "_weekday"
            try:
                df[c] = pd.to_datetime(df[c])
            except:  # noqa
                continue
            if len(self.columns) <= 1:
                col = 'weekday'
            df[col] = df[c].dt.day_name()
        return df


class SeasonProcessing(PreProcessing):
    """
    Extract season information from datetime columns in a DataFrame.

    Extracts season information from datetime columns in the data and stores the season
    names or numeric values in new column named either '<original name>_<season>', if
    multiple `columns` are declared else '<season>'.

    Parameters:
    :param columns: A list of column names containing datetime data from which seasons will be extracted.
                    (List[str])
    :param season: The type of season information to extract, either 'month' or 'quarter'.
                   Default is 'month'.
                   (str)
    :param naming: A boolean indicating whether to use season names (True) or numeric values (False).
                   Default is False.
                   (bool)

    Methods:
    :return: The DataFrame with season information extracted and stored in new columns.
             (pd.DataFrame)

    Example:
    >>> processor = SeasonProcessing(columns=['date_column'], season='month', naming=True)
    >>> df_processed = processor.transform(df)
    >>> print(df_processed.head())
    """
    def __init__(
        self,
        columns: List,
        season: str = 'month',
        naming: bool = False
    ):
        super().__init__(columns)
        self.season = season
        self.naming = naming

    def transform(self, df: pd.DataFrame):
        """
        Transform the DataFrame by extracting season information from datetime columns.

        This method converts the `df` columns to datetime objects, extracts season
        information (either names or numeric values), and creates new columns to store
        the season information.

        :param df: The DataFrame containing datetime columns to extract season information from.
                   (pd.DataFrame)
        :return: The transformed DataFrame with season information extracted and stored in new columns.
                 (pd.DataFrame)
        """
        for c in self.columns:
            col = '_'.join(c.split("_")[:-1]) + f"_{self.season}"
            try:
                df[c] = pd.to_datetime(df[c])
            except:  # noqa
                continue
            if len(self.columns) <= 1:
                col = self.season
            try:  # noqa
                if self.naming:
                    df[col] = getattr(df[c].dt, self.season+'_name')()
                else:
                    df[col] = getattr(df[c].dt, self.season)
            except Exception as e:
                logger.warn(e)
                continue
        return df


class ConTime(PreProcessing):
    """
    Build Continuous Time variable from integer.

    This pre-processing is forming a continuous variable, given either the declared
    `columns` or the default `time_col`.

    The output columns are sinusoids having their `peaks` at the desired position.
    E.g. Wednesday for a columns named 'weekday', then Wednesday it the peak.

    NOTE: Each period is modelled to have a peak value of *1.0* an the troph at *0.0* !

    :param columns: time columns to be processed
    :param peaks: dict with the desired peak of the sinusoid for each declared column
    :param time_col: default fallback column, if no (valid) time column has been declared
    """
    def __init__(
        self,
        columns: List[str],
        peaks: Dict[str, Union[int, float]] = None,
        time_col: Optional[str] = None
    ):
        super().__init__(columns)
        if peaks is None:
            peaks = {}
        if time_col is None:
            time_col = 'event_date'
        self.peaks = peaks
        self.time_col = time_col

    def infer_time(self, df, c):
        try:
            if 'datetime' not in df[self.time_col].dtype.__str__():
                df[self.time_col] = pd.to_datetime(df[self.time_col])
            col = getattr(df[self.time_col].dt, c)
        except AttributeError:
            col = None
        return col

    def transform(self, df: pd.DataFrame):
        for c in self.columns:
            if c not in df:
                logger.warn(f'Column {c} missing.'
                            f'Try infering time...')
                col = self.infer_time(df, c)
                if col is None:
                    logger.warn(f'Column {c} cannot be inferred! - skipping!')
                    continue
            else:
                col = df[c]
            if not pd.api.types.is_numeric_dtype(col):
                col = pd.to_numeric(col, errors='ignore')
                if not pd.api.types.is_numeric_dtype(col):
                    logger.warn(f'Column {c} is not numeric! '
                                f'Try infering time...')
                    col = self.infer_time(df, c)
                    if col is None:
                        logger.warn(f'Column {c} cannot be inferred! - skipping!')
                        continue
            peak = self.peaks.get(c, None)
            if peak is None:
                peak = np.max(col.unique()) + 1

            df[f'{c}_num_cont'] = col.map(
                lambda w: np.sin(2 * np.pi * w / peak)  # noqa
            ).tolist()
        return df


class GaussianDummy(PreProcessing):
    """
    Draw a gaussian around binary column space.

    uses :meth:`~modules.preprocessing.util import kgauss` function, which takes a k-dim
    one-hot encoded (OHE) input and draws a Gaussian bell kurve of the same size shifted
    around the corresponding active category.

    NOTE: Here, all of `columns` are considered to be the same OHE space!

    NOTE: The new columns will have the original names + '_gau' suffix!

    :param columns: List of the columns of the OHE space. (E.g. ['device_type_desktop', 'device_type_mobile', 'device_type_tablet']
    :param arange: List of the desired arrangement of the space.
    :param kwargs: Additional keyword args for :meth:`~modules.preprocessing.util import kgauss`

    """

    def __init__(
            self,
            columns: List[str],
            arange: Optional[Iterable],
            **kwargs
    ):
        super().__init__(columns)
        self.arange = arange
        self.kwargs = kwargs

    def transform(self, df: pd.DataFrame):
        """
        Transform `columns` from DataFrame.
        :param df: DataFrame containing all `columns`
        :return: DataFrame + new `columns` with suffix '_gau'
        """
        arange = self.arange
        if arange is None:
            arange = np.arange(len(self.columns)) - len(self.columns) // 2

        x = np.where(df[self.columns].values)[1]
        df[pd.Series(self.columns) + '_gau'] = pd.Series(x).map(
            lambda w: kgauss(
                np.array(arange), w - len(self.columns) // 2,
                self.kwargs.get('sigma', len(arange) * 1.1 / 7)
            )
        ).tolist()
        return df


class SpanTSPreprocessing(PreProcessing):

    def __init__(
            self,
            columns: List,
            groupers: List[str] = ['creative_id', 'advertizer'],  # noqa
            freq: Any = '1D',
            ts_start: Optional[float] = None,
            ts_end: Optional[float] = None
    ):
        super().__init__(columns=columns)
        assert len(columns) == 1
        self.groupers = groupers
        self.freq = freq
        self.ts_start = ts_start
        self.ts_end = ts_end

    def span_ts_partial(self, df, idx, col):
        gr, df = df
        df = pd.DataFrame(list(range(len(idx))), index=idx, columns=['ooo']).join(
            df.set_index(col), how='left'
        ).fillna(0).drop(
            ['ooo'], axis=1
        ).reset_index(
            drop=False
        ).rename(
            columns={'index': self.columns[0]}
        )
        df[self.groupers] = gr
        return df

    def transform(self, df: pd.DataFrame):
        if self.ts_end is None:
            self.ts_end = df[self.columns].max().max()
        if self.ts_start is None:
            self.ts_start = df[self.columns].min().min()
        df[self.columns[0]] = pd.to_datetime(df[self.columns[0]])
        idx = pd.date_range(
            start=self.ts_start,
            end=self.ts_end,
            freq=self.freq
        )
        with multiprocessing.Pool(NCPU) as pool:
            dfs = pool.map(
                partial(self.span_ts_partial, idx=idx, col=self.columns[0]),
                df.groupby(self.groupers)
            )
        df = pd.concat(dfs, axis=0)
        return df


class CosPeriod(Factorize):
    """
    Periodical Factorization.

    Uses a cosine with period of `len(columns)` with peak at `mean` as a factor model.

    NOTE: All `columns` are considered to be part of the same period space!

    :param columns: List of column names from the desired period.
    :param mean: value of where the peak of the factors should be.
    :param bias: Bias / minimum for the factorization.
    :param normalize_factors: Whether to normalize the fastors.
    :param bias_in_factors: Whether to include bias with the normalization.
    :param default_on_missing: Default factor on missing values.
    """
    def __init__(
        self,
        columns: list,
        mean: int = None,
        bias: float = None,
        normalize_factors: bool = True,
        bias_in_factors: bool = True,
        default_on_missing: float = 0
    ):
        k = len(columns)
        if mean is None:
            mean = k // 2
        arr = np.arange(k) - mean
        factor = (np.cos(2 * np.pi * arr / k) + 1) / k
        super().__init__(
            columns,
            factor,
            bias,
            normalize_factors,
            bias_in_factors,
            default_on_missing,
            '_cos'
        )
