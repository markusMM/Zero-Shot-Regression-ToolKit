import json
from datetime import datetime as dt
import os
import sys
import time
import traceback
import glob

from catboost import CatBoostRegressor
from sklearn.metrics import r2_score as r2
import numpy as np
from dask import dataframe as ddf
import pandas as pd
from modules.data_retrieval import s3_drvr as s3
from modules.utils import get_obj
from modules import model as MLModels
from modules import pipeline as MLPipes
from modules.log import logger
from modules.hpo.tuning_utils import tune_cp_model_skopt
from modules.data_retrieval.df_handlers import unstring_all
from sklearn.multioutput import MultiOutputRegressor
from modules.common import train_dtypes

S3_PIPE_BUFFER_SIZE = 500222333  # 500+ MB bytes
PROCESS_CHUNK_SIZE = 100444333  # 100+ MB bytes
np.int = np.int32


def process_input_data(
    df: pd.DataFrame,
    features: dict,
    targets: dict,
    # cosine_dist_features: list,
) -> pd.DataFrame:
    """
    Get Data Piped input from s3 into modules in chunks. Preporcess
    each chunk and concatenate all chunks. Final output is a dataframe.
    :param df: dataframe
    :features: dictionary of feature columns to be used assigned to their
    corresponding data type
    :targets: dictionary of target columns to be used assigned to their
    corresponding data type
    :return: Pandas Dataframe after converting input into final preprocessed
    dataframe.
    """
    logger.debug("processing data")
    df = unstring_all(df)

    # casting datatypes
    for column in df.columns:
        if column in features.keys():
            try:
                df[column] = df[column].astype(features[column])
            except Exception:
                logger.warning(f"Cannot cast {column} to {features[column]}:\n{column}")
        if column in targets.keys():
            try:
                df[column] = df[column].astype(targets[column])
            except Exception:
                logger.warning(f"Cannot cast {column} to {targets[column]}:\n{column}")
    logger.debug("done")
    return df


def grouped_training(
    df,
    model_class,
    groupers,
    xcols,
    ycols,
    train_ids,
    valid_ids,
    *model_args,
    **model_kwargs
) -> tuple:
    """
    Train multiple models based on some grouping.

    This function groups the data and trains and validates one estimator for each group.

    :param df: The input data. NOTE: Must contain all `groupers`, `xcols` and `ycols`!
    :param model_class: The estimator class. NOTE: Cannot be a `ModelWrapper`
    :param groupers: The columns to group by. NOTE: Need to have at least 1 entry!
    :param xcols: The input feature columns.
    :param ycols: The output target columns.
    :param train_ids: The indices of the rows to be used in training.
    :param valid_ids: The indices of the rows to be used in validation.
    :param model_args: Additional arguments for the estimator initialization.
    :param model_kwargs: Additional keyword arguments for the estimator initialization.
    :return (models, tr_scores, va_scores):
    A tuple of three dictionaries with models and scores and the group names as keys.
    """
    df['train'] = False
    df.loc[train_ids, 'train'] = True
    df['valid'] = False
    df.loc[valid_ids, 'valid'] = True
    dfs = df.groupby(groupers)
    models = {}
    tr_scores = {}
    va_scores = {}
    for g, df in dfs:
        try:
            train_ids = df.index[df['train']]
            valid_ids = df.index[df['valid']]
            model = MultiOutputRegressor(model_class(*model_args, **model_kwargs)).fit(
                df.loc[train_ids, xcols], df.loc[train_ids, ycols],
            )
            tr_scr = model.score(df.loc[train_ids, xcols], df.loc[train_ids, ycols])
            va_scr = model.score(df.loc[valid_ids, xcols], df.loc[valid_ids, ycols])
            logger.info(f'group {g} train R2: {tr_scr}')
            logger.info(f'group {g} valid R2: {va_scr}')
            models[g] = model
            tr_scores[g] = tr_scr
            va_scores[g] = va_scr
        except Exception as e:
            logger.warn(e)
            logger.warn(f'Cannot train the model on group "{g}"')
            models[g] = None
            continue
    return models, tr_scores, va_scores


def load_data_from_path(data_path):
    if '.csv' not in data_path:
        try:
            if train_dtypes is None:
                logger.warning(
                    'No training data types provided!'
                    'It is highly recommended to provide data types '
                    'for all relevant training columns!'
                )
            df = ddf.read_csv(
                data_path + '*.csv',
                dtype=train_dtypes
            ).compute()
        except Exception as e:  # noqa
            logger.warn(e.__traceback__)
            logger.warn(e)
            if 's3://' in data_path:
                bucket_name, subdir, sub = list(s3.parse_path(
                    data_path
                ).values())
                csvs = (
                    f's3://{bucket_name}/' +
                    pd.Series(s3.list_obj(bucket_name, subdir, sub))
                ).tolist()
            else:
                csvs = glob.glob(data_path + '/*.csv')
            df = pd.concat(list(map(
                lambda csv: pd.read_csv(csv),
                csvs
            )), axis=0)
    else:
        df = pd.read_csv(data_path)

    return df


# The function to execute the training.
def train(
    model_class: type = CatBoostRegressor,
    model_params: dict = dict(
        iterations=7000,
        metric_period=10,
        thread_count=-1,
        bootstrap_type='Bayesian',
        langevin=True,
        posterior_sampling=True,
        objective='RMSE',
        eval_metric="R2"
    ),
    train_data_path: str = "s3://cm-forecasting/data/processed_agg_native/train/",
    valid_data_path: str = None,
    train_idx: list = None,
    valid_idx: list = None,
    features: list = None,
    targets: list = None,
    pp_path: str = None,
    hpo_space: str = None,
    hpo_out_path: str = None,
    hpo_n_steps: int = 60,
    hpo_n_init: int = 30,
    fill_booleans_regex: list = [
        'goal_type_',
        'device_type_',
        'os_targets_',
        'browser_targets_'
        'language_',
        'ad_types_'
    ],
    **kwargs
) -> bool:
    """
    Train modules and write it to s3.

    Return true if successful else exit.

    :param train_data_path: String.
    path to training data
    :param valid_data_path: String.
    path to validation data (optional)
    :param model_path: String.
    path where the model artifacts will be stored
    :param output_path: String.
    path where the training logs will be written
    :param hyperparameter_file: String,
    path to the file where the hyperparamters are written.
    :return:
    """
    output_path = './opt/output'

    logger.info("loading data...")
    logger.info(train_data_path)

    # load data
    df_train = load_data_from_path(train_data_path)

    if (valid_data_path is not None) and (valid_data_path == valid_data_path):
        # load data
        df_valid = load_data_from_path(valid_data_path)
    else:
        if valid_idx is None and train_idx is None:
            N = len(df_train)
            idx = np.arange(N).tolist()
            df_train.index = idx
            np.random.shuffle(idx)
            train_idx = idx[:-N//5]
            valid_idx = idx[-N//5:]
        if valid_idx is not None:
            df_valid = df_train.loc[valid_idx]
        else:
            if len(train_idx) == len(df_train):
                train_idx = np.where(train_idx).tolist()
            df_valid = df_train.loc[set(train_idx).difference(df_train.index.tolist())]
        if train_idx is not None:
            df_train = df_train.loc[train_idx]
        else:
            if len(valid_idx) == len(df_train):
                valid_idx = np.where(valid_idx).tolist()
            df_train = df_train.loc[set(train_idx).difference(df_train.index.tolist())]

    # refill binary columns, if all false
    for fx in fill_booleans_regex:
        cols = list(filter(lambda c: fx in c, df_train.columns))
        if len(cols) > 0:
            df_train.loc[df_train[cols].values.sum(1) == 0, cols] = True
            if df_valid is not None:
                df_valid.loc[df_valid[cols].values.sum(1) == 0, cols] = True

    logger.info('done!')

    logger.info('loading pre-processing...')
    if pp_path is None:
        pp = s3.load_from('cm-forecasting', 'preprocessing/agg', 'ppax.pkl')
    else:
        try:
            pp = get_obj(pp_path)
            assert pp is not None
        except Exception as e:  # noqa
            pp = s3.load_from('cm-forecasting', 'preprocessing/agg', 'ppax.pkl')
    logger.info('done!')
    time_data = time.process_time()

    logger.info('starting training...')
    try:
        model = None
        if hpo_space is not None:
            if hpo_out_path is None:
                t_string = '_'.join(targets)
                hpo_out_path = f's3://cm-forecasting/hpo/{t_string}/{dt.now().timestamp()}'
            logger.info('found scikit-opt config:')
            logger.info(hpo_space)
            logger.info('tuning for best hyper parameters....')
            if df_valid is not None:
                df_train = df_train.reset_index()
                df_valid.index = np.arange(len(df_valid)) + df_train.index.values.max()  # noqa
                train_idx = df_train.index.tolist()
                valid_idx = df_valid.index.tolist()
                df_train = pd.concat([
                    df_train,
                    df_valid
                ], axis=0)
            _, model = tune_cp_model_skopt(
                model_class,
                df_train[features],
                df_train[targets],
                hpo_space,
                model_params,
                train_idx,
                valid_idx,
                hpo_out_path,
                hpo_n_steps,
                hpo_n_init,
                False,
                True
            )
        if model is None:
            if 'catboost' in str(model_class).lower() and df_valid is not None:
                model_obj = model_class(**model_params).fit(
                    df_train[features].fillna(0),
                    df_train[targets].fillna(0),
                    eval_set=(
                        df_valid[features].fillna(0),
                        df_valid[targets].fillna(0),
                    )
                )
                model_obj._random_seed = 42
            else:
                model_obj = model_class(**model_params).fit(
                    df_train[features].fillna(0),
                    df_train[targets].fillna(0)
                )
                model_obj.random_state = 42

            model = MLModels.SciKitRegressor(
                feature_columns=features,
                target_columns=targets,
                model_class=model_class,
                **model_params
            )
            model.model = model_obj

        if df_valid is not None:
            r2_valid = r2(model.predict(df_valid).values, df_valid[targets].values)
        else:
            Ntr = len(df_train)
            idx = df_train.index.tolist()
            np.random.shuffle(idx)
            va_ids = idx[:Ntr//5]
            r2_valid = r2(
                df_train.loc[va_ids][targets].values,
                model.predict(df_train.loc[va_ids]).values
            )
            mse_valid = ((
                df_train.loc[va_ids][targets].values -
                model.predict(df_train.loc[va_ids]).values
            )**2).mean(0)

        print(f'validation MSE: {mse_valid}')
        logger.info(f'validation MSE: {mse_valid}')

        time_train = time.process_time()
        logger.info(
            "We need {:f} seconds to train the model".format(
                time_train - time_data
            )
        )

        model_pipe = MLPipes.SkOptRecPipe(
            model_a=model,
            preprocessings_a=pp,
            features_a=features,
            targets_a=targets
        )

        model_pipe.r2_score = r2_valid
        model_pipe.score = r2_valid
        model_pipe.valid_score = r2_valid

        return model

    except Exception as e:
        # Write out an error file.
        # This will be returned as the failureReason in the
        # DescribeTrainingJob result.
        trc = traceback.format_exc()
        with open(os.path.join(output_path, "failure"), "w") as s:
            s.write("Exception during training: " + str(e) + "\n" + trc)
        # Printing the exception adapts it to be into the training job logs.
        logger.info("Exception during training: " + str(e) + "\n" + trc)
        # A non-zero exit code causes the training job to be marked as Failed.
        sys.exit(255)
