from modules.preprocessing.pre_processing import PreProcessing
import pandas as pd
import numpy as np
from functools import partial
import multiprocessing
from typing import Iterable
from modules.common import NCPU


class CountriesFromRegions(PreProcessing):
    """
    Fill or create a country targeting by some region targeting.

    This preprocessing grabs a region targeting column and references to the corre-
    sponding country.
    This is done once for each group, determined by `grouper`.

    NOTE: Only each first element in the group counts!
    So, consider to have a grouper which results in homogenous groups for all columns.

    :param columns: columns to be processed
    :param grouper: grouping to consider. Default 'profile_id'
    """

    def __init__(
            self,
            columns=[
                'country_targets',
                'region_targets'
            ],  # noqa
            grouper='profile_id'
    ):
        super().__init__(columns)
        self.grouper = grouper
        self.reg_col = columns[1]
        self.cnt_col = columns[0]

    def get_countries_from_regions_in_profile(
        self, df, region_df=None, country_df=None
    ):
        def get_country_from_region(region):
            if region is None:
                return ""
            code = region_df[region_df['region'] == region].get('country_code',
                                                                []).values
            if len(code) < 1:
                return ""
            code = code[0]
            country = country_df[country_df['code'] == code].get('country', []).values
            if len(country) < 1:
                return ""
            return country[0]

        prid, df = df
        regions = df[self.reg_col].values[0]
        if regions != regions or not isinstance(regions, Iterable):  # noqa
            countries = np.nan
        else:
            countries = list(set(np.unique(list(map(
                lambda r: get_country_from_region(r), regions
            ))).tolist()).difference(''))  # noqa

        return pd.DataFrame([[prid, countries]], columns=[
            self.grouper, 'region_country'
        ])

    def transform(self, df: pd.DataFrame) -> pd.DataFrame:

        def isna_or_empty(na):
            if isinstance(na, list):
                if len(na) < 1:
                    return True
            else:
                if na == np.nan:
                    return True
                if na is None:
                    return True
            return False

        with multiprocessing.Pool(NCPU) as pool:
            profiles = pool.map(
                partial(
                    self.get_countries_from_regions_in_profile,
                    region_df=REGION_DF,
                    country_df=COUNTRY_DF
                ),
                df.groupby(self.grouper)
            )
        profiles = pd.concat(profiles, axis=0)
        df = df.merge(profiles, on=self.grouper)
        if self.cnt_col in df:
            missings = df[self.cnt_col].apply(
                lambda x: isna_or_empty(x)
            ).values | df[self.cnt_col].isna().values
            df.loc[missings, self.cnt_col] = df.loc[missings, 'region_country']
        else:
            df[self.cnt_col] = df['region_country']
        return df
