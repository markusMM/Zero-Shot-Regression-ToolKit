import pandas as pd
import numpy as np
from functools import partial


def calc_filtered_vals(df, col, val_cols, filter_values, fun):
    result_df = pd.DataFrame(index=filter_values)

    for val_col in val_cols:
        distro_col = val_col + '_distro'
        result_df[distro_col] = 0  # Initialize all values to zero

        for filter_value in filter_values:
            mask = df[col].apply(
                lambda x: filter_value in x if isinstance(x, list) else False
            )
            filtered_values = fun(df.loc[mask, val_col].astype(np.float64))
            result_df.loc[filter_value, distro_col] = np.nanmean(filtered_values)

    return result_df


def make_distro_vals(df, col, val_cols, filter_values):
    df = calc_filtered_vals(df, col, val_cols, filter_values, np.nansum)
    return df / np.nansum(df, axis=0)


def make_rate_vals(df, col, val_col, filter_values, target):
    return calc_filtered_vals(
        df,
        col,
        [val_col],
        filter_values,
        partial(rate_val, target=target)
    )


def rate_val(rates, target):
    return np.nanmean((rates - target) / target)
