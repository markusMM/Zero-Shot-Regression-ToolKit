import logging
from datetime import datetime as dt
from json import JSONDecodeError
import numpy as np
import pandas as pd
import re
import json
from typing import List, Iterable
from modules.data_retrieval.media_retrieval import filter_video_list
import torch
nan = np.nan
array = np.array
tensor = torch.tensor


def save_unstring_elem(x: str):
    if not isinstance(x, str):
        return x
    try:
        return json.loads(x)
    except json.JSONDecodeError:
        try:
            return eval(x)
        except Exception as e:
            if ',' in x:
                try:
                    x = x.replace('[', '').replace(']', '')
                    x = x.replace(' ', '')
                    return x.split(',')
                except Exception:
                    pass
            logging.debug(e.__traceback__)
            logging.debug(e)
            logging.debug(f'Cannot translate string element: {x}')
            return x


def save_unstring_series(se: pd.Series):
    return se.apply(save_unstring_elem)


def save_unstring_all(df: pd.DataFrame) -> pd.DataFrame:
    df.loc[:, df.dtypes == 'O'] = df.loc[:, df.dtypes == 'O'].apply(
        save_unstring_series, axis=1
    )
    return df


def loads_list_str(df: pd.DataFrame, columns: List[str]) -> pd.DataFrame:
    for c in columns:
        df[c] = df[c].apply(
            lambda x: filter_video_list(json.loads(x))
            if type(x) is str
            else x
            if x is not None
            else None
        )
    return df


def select_first(df: pd.DataFrame, columns: List[str]) -> pd.DataFrame:
    for c in columns:
        df[c] = df[c].apply(lambda x: x[0] if x is not None else None)
    return df


def add_data(
    df1: pd.DataFrame,
    df2: pd.DataFrame,
    merge_cols: Iterable,
    time_col: str = None,
    time_res: str = None
) -> pd.DataFrame:
    """
l
    ..example:
    df_merged = add_data(df, df_reach, ['creative_id', 'event_date'], 'event_date')

    :param df1: dataframe the other table has to be added to
    :param df2: data which has to be added
    :param merge_cols: the index columns to merge the data from
    :param time_col: a possible datetime column in the merge columns
    :param time_res: resolver for the time resolution, eg date, year, etc
    :return: df1 appended by df2
    """
    if time_res is None:
        time_res = 'date'
    if not isinstance(merge_cols, list):  # noqa
        merge_cols = list(merge_cols)
    if time_col is not None:
        df1[time_col] = pd.to_datetime(getattr(df1[time_col].dt, time_res))
        df2[time_col] = pd.to_datetime(getattr(df2[time_col].dt, time_res))
        df1[time_col] = df1[time_col].apply(lambda d: int(d.timestamp()))
        df2[time_col] = df2[time_col].apply(lambda d: int(d.timestamp()))
    df1 = df1.set_index(merge_cols)
    df2 = df2.set_index(merge_cols)
    df2 = df2[list(filter(lambda c: c not in df1.columns, df2.columns))]
    try:
        df3 = df1.merge(df2, how='inner')
    except Exception as e:  # noqa
        print('Cannot merge!')
        print(e)
        df3 = df1.copy()
    df1 = df1.reset_index(drop=False)
    df2 = df2.reset_index(drop=False)
    df3 = df3.reset_index(drop=False)
    if time_col is not None:
        df1[time_col] = pd.as_datetime(df1[time_col].apply(dt.fromtimestamp))
        df2[time_col] = pd.as_datetime(df2[time_col].apply(dt.fromtimestamp))
        df3[time_col] = df3[time_col].apply(dt.fromtimestamp)
    return df3


def get_li_ids(df: pd.DataFrame):
    def loads_li(x):
        if x is None:
            return np.nan
        if len(x) <= 0:
            return np.nan
        try:
            x = json.loads(x)
        except JSONDecodeError:
            return np.nan
        if x is None:
            return np.nan
        return list(map(lambda v: v["id"] if v is not None else np.nan, x))

    if "line_items" in df:
        df["LIs"] = df["line_items"].apply(lambda x: loads_li(x))
    return df


def unstring_all(df: pd.DataFrame) -> pd.DataFrame:
    for c in df.columns:
        if sum(~df[c].isna()) <= 0:
            continue
        if (df[c].dtype == 'object' or df[c].dtype == str) and type(
            df[c][~df[c].isna()].iloc[0]
        ) == str:
            try:
                df.loc[
                    ~df[c].isna(), c
                ] = df.loc[
                    ~df[c].isna(), c
                ].apply(
                    lambda x: eval(
                        re.sub('(\d+|\d+\.\d+|[\w|\d]+)(\,|\])', '"\\1"\\2', x)
                    ) if isinstance(x, str) else x
                )  # noqa
            except Exception as e:  # noqa
                logging.debug(
                    f"{c} was not being able to converted "
                    f"from dtype={df[c].dtype}:\n{e}"
                )
    return df


def parse_device_types(df, col: str = 'device_type'):
    df[col] = df[col].astype(str).str.replace(
        'phone|mediaplayer|mobile phones|mobiles', 'mobile', regex=True
    ).str.replace(
        'pc|desktops|laptops', 'desktop', regex=True
    ).str.replace(
        '![desktop|tablet|mobile]', '', regex=True
    ).str.replace(
        ',,', ',', regex=False
    ).str.replace(
        '[,', '[', regex=False
    ).map(
        lambda dev: eval(dev) if dev is not None else np.nan
    )
    return df


def parse_target_names(
    df: pd.DataFrame,
    col: str,
    substrings: list = None,
    filters: list = None
) -> pd.DataFrame:
    def subs(name):
        for s in substrings:
            name = re.sub(s, '', name)
        return name

    def filt(name):
        for f in filters:
            if f in name:
                return f
        return None

    df[col] = df[col].apply(lambda t: list(filter(
        lambda n: n is not None,
        map(
            lambda b: filt(subs(b)), t
        )
    )))
    return df


def parse_browser_targets(
    df: pd.DataFrame,
    col: str = 'browser_targets',
    browsers: list = [
        'Internet Explorer',
        'Edge',
        'Firefox',
        'Chrome',
        'Safari',
        'Opera',
        'iPhone',
        'Android',
        'Samsung'
    ]
) -> pd.DataFrame:
    return parse_target_names(
        df,
        col,
        substrings=[' \d+$', ' (\.+)$'],  # noqa
        filters=browsers
    )
