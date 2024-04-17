import json
import torch
import pandas as pd
import numpy as np
from typing import List, Iterable
import logging


def flatten_simple_json(df, target_column):
    """
    Flatten a nested column that contains a json
    into multiple colums where each column is one field of the json.
    Example:
        |      json          |
        ----------------------
        | { "a": 1, "b": 2 } |
        | { "a": 2, "b": 3 } |

         result:
        | a | b |
        ----------
        | 1 | 2 |
        | 2 | 3 |
    """
    df[target_column] = df[target_column].fillna("{}")
    tmp = df.apply(lambda row: json.loads(row[target_column]), axis=1)
    return pd.concat([df.drop(target_column, axis=1), pd.json_normalize(tmp)], axis=1)


def flatten_list_json(df, target_column):
    """
        Flatten a nested column that contains a list of json objects
        into multiple colums, where each column is one field of the json.
        Example:
            |              json              |
            ----------------------------------
            | [{ "a": 1, "b": 2 }, {"a": 3, "c": 4}] |
            | [{ "a": 2, "b": 3 }, {"a": 4, "c": 5}] |

             result:
            | a_0 | b_0 | a_1 | c_1 |
            -------------------------
            |  1  |  2  |  3  |  4  |
            |  2  |  3  |  4  |  5  |
        """
    df[target_column] = df[target_column].fillna("[]").str.replace("", "[]")

    num_json = df[target_column].apply(lambda row: len(json.loads(row)))
    max_num_json = max(list(num_json))

    tmp = df.apply(lambda row: json.loads(row[target_column]), axis=1)
    tmp = tmp.apply(
        lambda row: [row[i] if i < len(row) else {} for i in range(max_num_json)]
    )
    tmp = pd.DataFrame(
        tmp.tolist(), columns=[f"{target_column}_{i}" for i in range(max_num_json)]
    )

    tmp = [
        pd.json_normalize(tmp[column_name]).add_prefix(f"{column_name}.")
        for column_name in tmp.columns
    ]
    return pd.concat([df.drop(target_column, axis=1)] + tmp, axis=1)


def flatten_targeting_columns(
    df,
    # Training data columns that contain simple json columns
    simple_json_columns: list = None,
    # Training data columns that contain list json columns
    listed_json_columns: list = None,
) -> pd.DataFrame:
    """Try to flatten a bunch of JSON and listed JSON columns.

    As mentioned above,
    this function tries and
    throws warnings about columns
    it wasn't able to transform.

    :inputs:
        df, pd.DataFrame:
            The dataframe containing the corresponding data.
        SIMPLE_JSON_COLUMNS, LIST[str]:
            The list of all simple JSON structured columns.
        LIST_JSON_COLUMNS, LIST[str]:
            The list of all listed JSON structured columns.

    :outputs:
        df, pd.DataFrame:
            The inputted dataframe with columns transformed
            inplace which worked.

    """
    if simple_json_columns is None:
        simple_json_columns = [
            "valuation",
            "video_targets",
            "supply_strategies",
            "auction_event",
            "inventory_discovery",
            "template",
            "inventory_url_whitelist_settings",
        ]
    if listed_json_columns is None:
        listed_json_columns = ["country_targets", "budget_intervals", "custom_models"]
    for c in simple_json_columns:
        logging.debug(f"Simple flattening of {c}. . . ")
        try:
            df = flatten_simple_json(df, c)
        except Exception as e:
            logging.warning(f"{c} could not have been flattened as simple JSON:\n{e}")

    for c in listed_json_columns:
        logging.debug(f"List nested flattening of {c}. . . ")
        try:
            df = flatten_list_json(df, c)
        except Exception as e:
            logging.warning(f"{c} could not have been flattened as listed JSON:\n{e}")
    return df


def decifer_age_targets_row(x: dict) -> pd.DataFrame:
    """ Get age groups out of the `age_targets` column.

    This function basically just grabs `allow_unknown` and two lists:
    lows and highs, which are representing to min and the max
    of the age groups.
    They are sorted respectively!
    Thus:
        ```
        {
             'low': df['age_targets_lows'][0],
             'high': df['age_targets_highs'][0]
        } == x['ages']
        ```
        shall be true if `x['ages']` would only contain `high` and `low`.

    :params:
        x, dict :
            a dictionary/JSON of the following form:
            ```
            {
                'allow_unknown': <bool>,
                ages: <List[
                    {
                        'low': <int>,
                        'high': <int>
                    }
                ] | None>
            }
            ```
    :outputs:
        df, pd.DataFrame :
            a dataframe of the following form:
            ```
            {
                'age_targets_allow_unknown': <bool>,
                'age_targets_lows': <List[int]>,
                'age_targets_highs': <List[int]>
            }
            ```
    <note>
        This function is more or less depricated!
        There is an SQL query doing exactly this!
    </note>
    """
    df = pd.DataFrame(
        {
            "age_targets_allow_unknown": True,
            "age_targets_lows": [[]],
            "age_targets_highs": [[]],
        },
        index=[0],
    )
    if x is None:
        return df
    if isinstance(x, str):
        x = json.loads(x)
    if x == {}:
        return df
    df["age_targets_allow_unknown"] = x["allow_unknown"]
    if hasattr(df, "ages"):
        if x["ages"] is not None:
            df["age_targets_lows"] = [list(map(lambda a: a["low"], x["ages"]))]
            df["age_targets_highs"] = [list(map(lambda a: a["high"], x["ages"]))]
        else:
            df["age_targets_lows"] = [[]]
            df["age_targets_highs"] = [[]]
    else:
        df["age_targets_lows"] = [[]]
        df["age_targets_highs"] = [[]]
    return df


def decifer_age_targets(df: pd.DataFrame, get_age_df: bool = True):
    """Build rows for all lows and highs of ages in the targeting.

    <note>
        This function is more or less depricated!
        There is an SQL query doing exactly this!
    </note>

    does create columns:
    - `age_targets_allow_unknown`
    - `age_targets_lows`
    - `age_targets_highs`
    and appends them to the input dataframe,
    as long as `age_targets` are in that dataframe.

    :inputs:
        df, `pd.DataFrame`:
            the input dataframe
        get_age_df, bool:
            Whether we only want the created dataframe or the whole one.

    :outputs:
        df | df_ages, `pd.DataFrame`:
            Either the whole dataframe with the age dataframe appended
            or only the age one.
    """
    try:
        assert "age_targets" in df
        df_ages = pd.concat(
            df["age_targets"].apply(lambda r: decifer_age_targets_row(r)).tolist()
        )
        if get_age_df:
            return df_ages
        return pd.concat([df, df_ages])
    except AssertionError:
        logging.warning(f"age_targets missing in columns!\n{df.columns}")


def age_hist(
    low: List[int],
    high: List[int],
    normalize: bool = True,
    nbins: int = 12,
    h_low: int = 1,
    h_high: int = 120,
):
    """Make one histogram for an age group target.

    It spans a histogram between `h_low` and `h_high` with
    resolution of `nbins`,
    given all `low` and `high` values from the age groups,
    sorted correspondingly.
    Additionally, it normalizes then between 0 and 1 and
    optionally, it normalizes each histogram to sum up to 1,
    by setting `normalize` to `True`.
    (That option is usually much more accurate!)
    """
    hist = None
    if (low is None) or (high is None):
        hist = torch.ones((1, nbins))
    else:
        if (len(low) < 1) or (len(high) < 1):
            hist = torch.ones((1, nbins))
    if hist is None:
        low = np.array(low)
        high = np.array(high)
        hilo_tuple = np.concatenate([low[:, None], high[:, None]], axis=1)
        hist = torch.cat(
            list(
                map(
                    lambda lh: torch.arange(lh[0], lh[1] + 1)
                    .float()
                    .histc(nbins, h_low, h_high)[None]
                    / (h_high - h_low + 1)
                    * nbins,
                    hilo_tuple,
                )
            ),
            dim=0,
        ).sum(0)
    if normalize:
        hist /= hist.norm()
    return hist.detach().numpy()[None]


def make_age_histograms(
    df: pd.DataFrame,
    age_cols: list = None,
    normalize: bool = True,
    nbins: int = 12,
    h_low: int = 1,
    h_high: int = 120,
):
    """Make one histogram, per  instance, for the age group targets.

    It spans a histogram between `h_low` and `h_high`
    with resolution of `nbins`,
    given all `low` and `high` values from the age groups,
    sorted correspondingly.
    Additionally, it normalizes then between 0 and 1 and
    optionally, it normalizes each histogram to sum up to 1,
    by setting `normalize` to `True`.
    (That option is usually much more accurate!)
    """
    if age_cols is None:
        age_cols = ["age_targets_lows", "age_targets_highs"]
    assert len(age_cols) > 1
    df[[f"age_hist_{k}" for k in range(nbins)]] = np.concatenate(
        list(
            map(
                lambda j: age_hist(
                    df.iloc[j][age_cols[0]],
                    df.iloc[j][age_cols[1]],
                    normalize=normalize,
                    nbins=nbins,
                    h_low=h_low,
                    h_high=h_high,
                ),
                range(len(df)),
            )
        ),
        axis=0,
    )
    return df


def unstring_all(df: pd.DataFrame) -> pd.DataFrame:
    """Tries to JSON load all `JSON` object or `str` columns."""
    for c in df.columns:
        if sum(~df[c].isna()) <= 0:
            continue
        if (df[c].dtype == object or df[c].dtype == str) and type(
            df[c][~df[c].isna()].iloc[0]
        ) not in [list, dict, np.ndarray, set]:
            try:
                df[c] = df[c].apply(
                    lambda x: json.loads(x) if isinstance(x, str) else None
                )  # noqa
            except Exception as e:
                logging.debug(
                    f"{c} was not being able to "
                    f"converted from dtype={df[c].dtype}:\n{e}"
                )
    return df


def kgauss(x, mu=0, sigma=1.1):
    """
    Calculate the Gaussian function values for given input.

    Computes the values of a Gaussian function for a given input array, using the
    specified mean (mu) and standard deviation (sigma).

    :param x: Input array for which Gaussian function values will be computed.
              (np.ndarray)
    :param mu: Mean value of the Gaussian function. Default is 0.
               (float)
    :param sigma: Standard deviation of the Gaussian function. Default is 1.1.
                  (float)

    :return: Array of Gaussian function values corresponding to the input array.
             (np.ndarray)

    Example:

    >>> x_values = np.linspace(-5, 5, 100)
    >>> gaussian_values = kgauss(x_values, mu=0, sigma=1.5)
    >>> print(gaussian_values[:12])
    ... [0.00000000e+00 9.66203177e-04 2.14541791e-03 3.57747440e-03
    ...  5.30793657e-03 7.38854394e-03 9.87760179e-03 1.28402959e-02
    ...  1.63489056e-02 2.04828857e-02 2.53287876e-02 3.09799880e-02]
    """
    y_gauss = np.exp(-np.pi*sigma**2 - .5*(x - mu)**2 / sigma**2)
    y_gauss -= y_gauss.min()
    y_gauss /= y_gauss.max()
    return y_gauss


def deep_uniques(arr: Iterable):
    """ Dive into nested array and find all unique items. """
    return np.unique(np.concatenate(list(map(lambda x: x.squeeze(), arr))))


def svd_low_rank_pca(
    tensor: torch.Tensor, n_comp: int = None
) -> torch.Tensor:
    """
    Perform low-rank PCA using Singular Value Decomposition (SVD) on the input tensor.

    Computes low-rank PCA using Singular Value Decomposition (SVD) on the input tensor.
    It returns the right singular vectors (v), the pseudo-inverse of v, the singular
    values (s), and the left singular vectors (u).

    :param tensor: Input tensor for which low-rank PCA will be computed.
                   (torch.Tensor)
    :param n_comp: Number of principal components to use. If None, all components are used.
                   Default is None.
                   (int)

    :return: Tuple containing the right singular vectors (v), the pseudo-inverse of v,
             the singular values (s), and the left singular vectors (u).
             (Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor])

    Example:

    >>> data_tensor = torch.randn(100, 10)
    >>> v, v_pseudo_inv, s, u = svd_low_rank_pca(data_tensor, n_comp=5)
    >>> print(v.shape, v_pseudo_inv.shape, s.shape, u.shape)
    ... torch.Size([10, 5]) torch.Size([5, 10]) torch.Size([5]) torch.Size([100, 5])
    """
    if n_comp is None:
        n_comp = len(tensor)
    tensor = (tensor - tensor.mean(axis=0)) / tensor.std(dim=0)
    u, s, v = torch.pca_lowrank(tensor, n_comp)
    return v, torch.pinverse(v), s, u


def firstel(arr):
    """ Just returns the first element of an array. """
    if isinstance(arr, pd.Series):
        arr = arr.values
    arr = np.array(arr)
    return arr[0]
