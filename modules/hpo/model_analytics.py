import glob
import json
import re
from itertools import product
from typing import Union, Tuple, Callable
import joblib
import numpy as np
import pandas as pd
import torch
import shap
from datetime import datetime as dt
from forex_python import converter as cur_conv
from modules.inference_utils import translate_request
from modules.data_retrieval import s3_drvr as s3
from modules.log import logger
from modules.pipeline import PipeLine
from plotly import express as px


def shap_mlp_model(
        mini_data_path: str,
        model_path: str,
        plot_feature_importance: bool = True,
        deep: bool = False
) -> Union[pd.DataFrame, Tuple[pd.DataFrame]]:
    data = pd.read_csv(
        mini_data_path,
        index_col=0
    )
    if 's3://' in model_path:
        model_path = model_path.replace('s3://', '')
        bucket = model_path.split('/')[0]
        subdir = '/'.join(model_path.split('/')[1:-1])
        key = model_path.split('/')[-1]
        mlp_pl = s3.load_from(bucket, subdir, key)
    else:
        mlp_pl = joblib.load(model_path)

    xcols = mlp_pl.features

    shexp = shap.DeepExplainer(
        mlp_pl.model,
        torch.from_numpy(
            data[xcols].values.astype(float)
        ).float()
    )
    shap_values = shexp.shap_values(
        torch.from_numpy(
            data[xcols].values.astype(float)
        ).float()
    )
    if plot_feature_importance:
        shap.summary_plot(
            shap_values,
            features=data[xcols].values.astype(float),
            feature_names=xcols
        )
    shap_values = pd.DataFrame(
        shap_values,
        columns=xcols
    )

    if deep:
        raise NotImplemented

    return shap_values


def alias_brev(df: pd.DataFrame) -> pd.DataFrame:
    try:
        cur_rate = cur_conv.get_rate('EUR', 'USD', dt.now())
    except ConnectionError as e:
        logger.warn(e.__traceback__)
        logger.warn(e)
        logger.warn('Cannot draw exchange rate! Use default...')
        cur_rate = 1.0733
    df['booked_revenue_usd'] = df['budget'] * cur_rate
    return df


def case_test_model(
        model_path: str,
        test_data_root: str = '**/test_data',
        test_data_filt: list = ['fc_sample*.json'],
        pp_path: str = None,
        data_tf: Callable = alias_brev,
        reg_x_var: str = 'poly_booked_revenue_usd',
        reg_coefs: dict = {
            'poly_1': 0,
            'poly_booked_revenue_usd': 81.589082,
            'poly_booked_revenue_usd^2': 0.189342,
            'poly_booked_revenue_usd^3': -0.000104
        },
        reg_gens: dict = {
            'ploy_1': '1',
            'poly_booked_revenue_usd': 'x',
            'poly_booked_revenue_usd^2': 'poly_booked_revenue_usd**2',
            'poly_booked_revenue_usd^3': 'poly_booked_revenue_usd**3'
        },
        reg_threshold: float = 50_000.0,
        reg_cut_off: float = 1300
) -> dict:
    """
    Test series of different cases based on a sigleton linear relationsship.

    The initial relationship has to be a single variable.
    But the final variables for regression can be multiple.
    However, it is recommended to only use ablations of one variable.
    Additionally, the x-variable needs to be already given in the request patterns.

    The requests get transformed by `data_tf` if given.
    They also get pre-processed by a preprocessing given in `pp_path`.
    If `pp_path` is not given, it will be ignored!


    :param model_path:
    :param pp_path:
    :param data_tf:
    :param test_data_root:
    :param test_data_filt:
    :param reg_x_var:
    :param reg_coefs:
    :param reg_gens:
    :param reg_threshold:
    :param reg_cut_off:
    :return: Dictionary of resulting MAD, MSD, failure rate and no. total instances.
    """
    pp = None
    if reg_cut_off is None:
        reg_cut_off = np.inf

    if 's3//' in model_path:
        model_path = model_path.replace('s3//:', '')
        bucket = model_path.split('/')[0]
        subdir = '/'.join(model_path.split('/')[1:-1])
        key = model_path.split('/')[-1]
        model = s3.load_from(bucket, subdir, key)
    else:
        model = joblib.load(model_path)

    if pp_path is not None:
        try:
            if 's3//' in pp_path:
                pp_path = pp_path.replace('s3//:', '')
                bucket = pp_path.split('/')[0]
                subdir = '/'.join(pp_path.split('/')[1:-1])
                key = pp_path.split('/')[-1]
                model = s3.load_from(bucket, subdir, key)
            else:
                model = joblib.load(model_path)
        except Exception as e:  # noqa
            pp = None
            logger.warn(e.__traceback__)
            logger.warn(e)
            logger.warn(f'Could not load pp from path: {pp_path}')

    def reg_mod(
            x,
            x_var: str = reg_x_var,
            gens: dict = reg_gens,
            coefs: object = reg_coefs,
            cut: float = reg_cut_off
    ) -> np.ndarray:
        r_vars = gens.keys()
        x[x > cut] = cut
        dfx = pd.DataFrame(x[:, None], columns=[x_var])
        for c in set(gens.keys()).difference([x_var]):
            dfx[c] = eval(gens[c])
        beta = np.array([coefs[j] for j in r_vars])[None]
        return (dfx[r_vars] * beta).sum(1)

    test_jsons = []
    for r in test_data_filt:
        g = (glob.glob(
            f'{test_data_root}/{r}',
            recursive=True
        ) + glob.glob(
            f'../{test_data_root}/{r}',
            recursive=True
        ))
        test_jsons += g
    logger.info(f'stating tests with the following input patterns: {test_jsons}')
    mad = 0
    msd = 0
    fails = 0
    n_instance = 0
    for g in test_jsons:
        logger.info(f'loading {g}')
        try:
            with open(g) as f:
                req = translate_request(json.load(f))
        except Exception as e:  # noqa
            logger.warn(e.__traceback__)
            logger.warn(e)
            logger.warn(f'unable to load pattern {g}')
            continue
        if data_tf is not None:
            logger.info('transforming')
            try:
                df = data_tf(df)
                assert isinstance(df, pd.DataFrame)
            except Exception as e:  # noqa
                logger.info(e.__traceback__)
                logger.info(e)
                logger.info('Cannot transform data!')
        if pp is not None:
            logger.info('pre-processing...')
            for p in pp:
                req = p.transform(df)
        logger.info(f'Testing pattern: {req}')
        pred = model.predict(req)
        dif = np.abs(pred - reg)
        reg = reg_mod(req[reg_x_var])
        mad += np.nansum(dif)
        msd += np.nansum((pred - reg)**2)
        fails += np.nansum(dif > reg_threshold)
        n_instance += len(pred)
    mad /= n_instance
    msd /= n_instance
    logger.info('all tests done!')
    logger.info(f'fail rate: {fails / n_instance}')
    logger.info(f'MAD: {mad}')
    logger.info(f'MSD: {msd}')
    return {
        'MAD': mad,
        'MSD': msd,
        'fail_rate': {fails / n_instance},
        'fails': fails,
        'threshold': reg_threshold,
        'coefs': reg_coefs,
        'variable_of_interest': reg_x_var
    }


def catboost_feature_importances(
    cb_model,
    feature_names: list,
    feature_groups: dict = None,
    do_plot: bool = False
) -> pd.Series:
    # parsing and mapping
    feature_importance = pd.Series(
        cb_model.feature_importances_,
        index=feature_names,
        name='feature_importances'
    )
    feature_importance = feature_importance.sort_values(ascending=True)

    # grouping, iff given
    if feature_groups is not None:
        fi = dict()
        gcols = []
        for g in feature_groups:
            fi[g] = feature_importance.loc[feature_groups[g]].abs().sum()
            gcols.extend(feature_groups[g])
        # backfill ungrouped features
        for f in set(feature_importance.index).difference(gcols):
            fi[f] = feature_importance[f]
        feature_importance = pd.Series(
            fi, name='feature_importances'
        ).sort_values(ascending=True)

    # plot
    if do_plot:
        from matplotlib import pylab as plb
        k = len(feature_importance)
        fig, ax = plb.subplots(figsize=(20, k // 2))
        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)
        ax.spines['left'].set_visible(False)
        ax.grid(axis='x')
        ax.barh(
            range(k),
            feature_importance.values,
            align='center'
        )
        plb.yticks(
            range(k),
            np.array(feature_importance.index.values)
        )
        fig.tight_layout()
        plb.show()

    return feature_importance


def shap_cb(
    cb_model,
    df: pd.DataFrame,
    feature_names: list,
    feature_groups: dict = None,
    do_plot: bool = False,
    do_summary_plot: bool = False
) -> pd.Series:
    # parsing and mapping
    explainer = shap.TreeExplainer(cb_model)
    shap_values = explainer(df[feature_names])
    feature_names = shap_values.feature_names
    shap.summary_plot(shap_values)
    feature_effect = pd.DataFrame(
        shap_values.values, columns=feature_names
    ).mean(axis=0)

    # grouping, iff given
    if feature_groups is not None:
        fi = dict()
        gcols = []
        for g in feature_groups:
            fi[g] = feature_effect.loc[feature_groups[g]].sum()
            gcols.extend(feature_groups[g])
        # backfill ungrouped features
        for f in set(feature_effect.index).difference(gcols):
            fi[f] = feature_effect[f]
        feature_effect = pd.Series(
            fi, name='feature_importances'
        ).sort_values(ascending=True)

    # plot
    if do_plot:
        from matplotlib import pylab as plb
        k = len(feature_effect)
        fig, ax = plb.subplots(figsize=(20, k // 2))
        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)
        ax.spines['left'].set_visible(False)
        ax.grid(axis='x')
        ax.barh(
            range(k),
            feature_effect.values,
            align='center'
        )
        plb.yticks(
            range(k),
            np.array(feature_effect.index.values)
        )
        fig.tight_layout()
        plb.show()
        if do_summary_plot:
            shap.summary_plot(shap_values)

    return shap_values


def analyze_response_on_option(
        ml_pipe: PipeLine,
        df: pd.DataFrame,
        kpi: str = 'imps_viewed',
        fields: list = [
            {
                'regex': 'device_type_',
                'options': [True, False]
            },
            {
                'regex': 'os_',
                'options': [True, False]
            },
            {
                'regex': 'browser_targets_',
                'options': [True, False]
            },
            {
                'regex': 'weekday_',
                'options': np.arange(0, 2, 1).tolist()
            }
        ],
        normalized_fields: list = [
            'weekday_',
            'hour_'
        ],
        preprocessed_data: bool = True,
        keep_samples: bool = False,
        plot_samples: bool = False,
        plot_diff: bool = False
):
    assert isinstance(ml_pipe, PipeLine)
    df = df.copy()

    # fields loop
    all_stats = {}
    for f in fields:
        # variables
        rex = f['regex']
        norm = rex in normalized_fields
        options = f['options']
        fcols = list(filter(lambda c: rex in c, df.columns))
        fname = re.sub('_$', '', rex)
        op_names = pd.Series(
            fcols, name=fname
        ).str.replace(rex, '')

        # options loop
        stats = {}
        k = len(fcols)
        preds = []
        samples = []
        for op in product(options, repeat=k):
            # option parsing
            op = np.array(op)
            original = op
            if norm:
                op = op / np.nansum(op)

            # prepare dat
            df_ = df.copy()
            df_[fcols] = [op.tolist()]*len(df_)
            if preprocessed_data:
                pred = ml_pipe.model_a.predict(df_)
            else:
                pred = ml_pipe.predict(df_)
            preds.append(pred.values)
            samples.append(original)

        preds = np.stack(preds, 0).reshape(-1, pred.shape[-1])
        samples = pd.DataFrame(
            np.stack(samples, axis=0),
            columns=fcols
        )
        stats['var'] = pd.DataFrame(preds.var(0)[None], columns=pred.columns)
        stats['std'] = pd.DataFrame(preds.std(0)[None], columns=pred.columns)
        stats['mean'] = pd.DataFrame(preds.mean(0)[None], columns=pred.columns)

        if keep_samples:
            stats['samples'] = samples
            stats['predictions'] = preds

        all_stats[', '.join(fcols)] = stats

        if plot_samples:
            def make_op_list(d: pd.Series):
                if d.dtype == 'bool':
                    d = d.astype(int)
                return d.tolist()

            samples_compr = pd.DataFrame(samples.apply(
                lambda d: f'{make_op_list(d)}',
                axis=1
            ), columns=[fname + ' (' + str(op_names.tolist()) + ')']).copy()
            px.scatter(
                pd.concat([
                    pd.DataFrame(preds, columns=pred.columns),
                    samples_compr
                ], axis=1),
                x=fname + ' (' + str(op_names.tolist()) + ')',
                y=kpi
            ).show()

    return stats
