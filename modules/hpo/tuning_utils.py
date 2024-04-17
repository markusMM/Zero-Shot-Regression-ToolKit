import os
import glob
import joblib
from typing import Union
from datetime import datetime as dt
import numpy as np
import pandas as pd
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import r2_score as r2
from skopt import space, gp_minimize
from skopt.utils import use_named_args
from torch.nn import PoissonNLLLoss, L1Loss
from torch.optim.lr_scheduler import CosineAnnealingLR
from catboost import CatBoostRegressor
from distutils.dir_util import mkpath
from modules.data_retrieval.s3_drvr import dump_to, list_obj
from modules.data_retrieval import s3_drvr as s3
from modules.statistics.stat_div import compute_js_divergence
from modules.model.scikit_model import SciKitRegressor
from modules.model.torch_model import LightningModel
from modules.model.algos.neural_net import MLP
from modules.model.algos.layers import nn
from modules.log import logger
from modules.common import NCPU
from modules.utils import get_obj
import torch

# hotfix for missing int alias in Numpy
np.int = np.int32


def get_max_runs(checkpoint_path):
    if 's3:' not in checkpoint_path.lower():
        mkpath(checkpoint_path)
        bucket_name = None
        bucket_dir = None
    else:
        bucket_name = checkpoint_path.split('s3://')[1].split('/')[0]
        bucket_dir = '/'.join(checkpoint_path.split('s3://')[1].split('/')[1:])
    if 's3:' not in checkpoint_path.lower():
        run_list = glob.glob(checkpoint_path + '/cb_run_*.pkl')
    else:
        run_list = list_obj(bucket_name, bucket_dir, '\_\d+.pkl')  # noqa
    runs = list(map(
        lambda r: int(r.split('_')[-1].split('.')[0]),
        run_list
    ))
    if len(runs) > 0:
        max_run = max(runs)
    else:
        max_run = 0
    return max_run


def check_run_mtr(obj_path: str) -> str:
    if 's3:' not in obj_path.lower():
        obj = joblib.load(obj_path)
    else:
        obj = s3.load_from(**s3.parse_path(obj_path))
    mtr = getattr(
        obj, 'targets',
        getattr(
            obj, 'target_columns',
            getattr(obj, 'target_cols', [])
        )
    )
    return mtr


def check_run_metrics(checkpoint_path):
    if 's3:' not in checkpoint_path.lower():
        mkpath(checkpoint_path)
        bucket_name = None
        bucket_dir = None
    else:
        bucket_name = checkpoint_path.split('s3://')[1].split('/')[0]
        bucket_dir = '/'.join(checkpoint_path.split('s3://')[1].split('/')[1:])
    if 's3:' not in checkpoint_path.lower():
        run_list = glob.glob(checkpoint_path + '/cb_run_*.pkl')
    else:
        run_list = pd.Series(
            list_obj(bucket_name, bucket_dir, '\_\d+.pkl')  # noqa
        )
        run_list = ('s3://' + bucket_name + '/' + run_list).tolist()
    run_metrics = list(map(
        lambda r: check_run_mtr(r),
        run_list
    ))
    return run_metrics


def predict_variance_test(model, df):
    minmax = MinMaxScaler()
    get_minmax_var = lambda x: (minmax.fit_transform(x)).var(axis=0)  # noqa
    vx = get_minmax_var(df[model.features].values)
    vy = get_minmax_var(df[model.targets].values)
    vp = get_minmax_var(model.predict(df).values)
    dv_yp = pd.Series(
        np.abs(vy - vp) / vy,
        index=model.targets
    )
    dv_xp = pd.DataFrame(
        np.abs(vx[:, None] - vp[None]) / vx[:, None],
        index=model.features,
        columns=model.targets
    )
    return {
        'feature2prediction_variance': dv_xp,
        'target2prediction_variance': dv_yp
    }


def calc_metrics(
        nn_model,
        df: pd.DataFrame,
        y_columns: list = ['log_imps_viewed'],
        train_ids: list = None,
        valid_ids: list = None
) -> dict:
    metrics = {
        'va_scr': nn_model.score(df.loc[valid_ids].fillna(0)),
        'tr_scr': nn_model.score(df.loc[train_ids].fillna(0))
    }
    pvar_test = predict_variance_test(nn_model, df.loc[valid_ids].fillna(0))
    metrics['mad_x_variance'] = np.nanmean(pvar_test['feature2prediction_variance'])
    metrics['mad_y_variance'] = np.nanmean(pvar_test['target2prediction_variance'])
    tr_p = torch.tensor(
        nn_model.predict(df.loc[train_ids].fillna(0)).values
    ).float()
    va_p = torch.tensor(
        nn_model.predict(df.loc[valid_ids].fillna(0)).values
    ).float()
    tr_y = torch.tensor(df.loc[train_ids, y_columns].fillna(0).values).float()
    va_y = torch.tensor(df.loc[valid_ids, y_columns].fillna(0).values).float()
    tr_nozero = np.where(tr_y != 0)[0].tolist()
    va_nozero = np.where(va_y != 0)[0].tolist()
    metrics['va_PNLL'] = PoissonNLLLoss()(
        va_p,
        va_y
    )
    metrics['tr_PNLL'] = PoissonNLLLoss()(
        tr_p,
        tr_y
    )
    metrics['tr_non_zero_mae'] = L1Loss()(tr_p[tr_nozero], tr_y[tr_nozero]).item()
    metrics['va_non_zero_mae'] = L1Loss()(va_p[va_nozero], va_y[va_nozero]).item()
    metrics['tr_non_zero_js_div'] = compute_js_divergence(
        tr_p[tr_nozero], tr_y[tr_nozero], n_bins=96
    ).item()
    metrics['va_non_zero_js_div'] = compute_js_divergence(
        va_p[va_nozero], va_y[va_nozero], n_bins=96
    ).item()
    metrics['tr_mae'] = np.nanmean(np.abs(tr_p - tr_y))
    metrics['va_mae'] = np.nanmean(np.abs(va_p - va_y))
    metrics['tr_mad_acc'] = 1 - metrics['tr_mae'] / tr_y.std().item()
    metrics['va_mad_acc'] = 1 - metrics['va_mae'] / va_y.std().item()
    return metrics


def tune_lightning_mlp_architecture(
        df: pd.DataFrame,
        train_ids: list,
        valid_ids: list,
        x_columns: list = None,
        y_columns=None,
        checkpoint_path='./models/hpo',
        n_hidden_variants=None,
        use_variable_adbudg: bool = False,
        use_variable_dropout: bool = False,
        use_variable_batchnorm: bool = False,
        use_variable_layernorm: bool = True,
        use_response_adbudg: bool = True,
        use_response_dropout: bool = False,
        use_response_batchnorm: bool = True,
        use_response_sigmoid: bool = False,
        use_skip_connections: bool = False,
        batch_size: int = 500,
        t_max: int = 220,
        t_burn: int = 150,
        t_ann: int = 200,
        loss_fun: str = 'L1Loss',
        constraint_weight: float = .3,
        constraint_loss_model: Union[object, str] = None,
        constraint_x_cols: list = None,
        lr_init: float = 0.1,
        return_best_model: bool = False,
        model_desc_mtr: str = None,
        model_desc_fun: str = 'min',
        train_flag: bool = False,
        del_lightning_logs: bool = True
) -> list:
    # make path
    if not train_flag:
        checkpoint_path = os.path.join(checkpoint_path, f'{dt.now().timestamp()}')
    if 's3:' not in checkpoint_path.lower():
        mkpath(checkpoint_path)
        bucket_name = None
        bucket_dir = None
    else:
        bucket_name = checkpoint_path.split('s3://')[1].split('/')[0]
        bucket_dir = '/'.join(checkpoint_path.split('s3://')[1].split('/')[1:])

    # default param parsing
    if n_hidden_variants is None:
        n_hidden_variants = [
            [200, 600],
            [600],
            [600, 800],
            [1024]
        ]
    if y_columns is None:
        y_columns = ['log_imps_viewed']
    if model_desc_mtr is None:
        model_desc_mtr = 'va_non_zero_mae'
    if model_desc_fun is None:
        model_desc_fun = 'min'

    # ADN (active, dropout and norm) parsing
    model_adns = dict(
        use_variable_adbudg=use_variable_adbudg,
        use_variable_dropout=use_variable_dropout,
        use_variable_batchnorm=use_variable_batchnorm,
        use_variable_layernorm=use_variable_layernorm,
        use_response_adbudg=use_response_adbudg,
        use_response_dropout=use_response_dropout,
        use_response_batchnorm=use_response_batchnorm,
        use_response_sigmoid=use_response_sigmoid
    )
    input_adn = ''.join(list(map(
        lambda f:
        f + ' ' if model_adns.get(f'use_variable_{f}', False) else '',
        [
            'batchnorm',
            'layernorm',
            'dropout',
            'adbudg'
        ]
    )))
    output_adn = ''.join(list(map(
        lambda f:
        f + ' ' if model_adns.get(f'use_response_{f}', False) else '',
        [
            'batchnorm',
            'dropout',
            'adbudg',
            'sigmoid'
        ]
    )))
    valas = []
    if 's3://' not in checkpoint_path.lower():
        run_list = glob.glob(checkpoint_path + '/mlp_run_*.pt')
    else:
        run_list = list_obj(bucket_name, bucket_dir, '\_\d+.pkl')  # noqa
    runs = list(map(
        lambda r: int(r.split('_')[-1].split('.')[0]),
        run_list
    ))
    if len(runs) > 0:
        max_run = max(runs)
    else:
        max_run = 0
    best_mdl = None
    best_mtr = None
    for j, n_hidden in enumerate(n_hidden_variants):
        mlp_pl_cr = LightningModel(
            feature_columns=x_columns,
            target_columns=['log_imps_viewed'],
            model_class=MLP,
            model_args=dict(
                n_input=len(x_columns),
                n_output=1,
                n_hidden=n_hidden,
                skip_net=use_skip_connections,
                lr_init=lr_init,
                loss=getattr(nn, loss_fun, getattr(locals(), loss_fun, nn.MSELoss))(),
                seed=42,
                constraint_loss_model=constraint_loss_model,
                constraint_weight=constraint_weight
            ) | model_adns,
            scheduler=CosineAnnealingLR,
            scheduler_params=dict(
                T_max=t_ann,
                eta_min=1e-7
            ),
            train_idx=train_ids,
            valid_idx=valid_ids,
            max_epochs=t_max,
            burn_in_steps=t_burn,
            batch_size=batch_size,
            constraint_x_cols=constraint_x_cols,
        ).fit(df.fillna(0))
        if del_lightning_logs:
            if os.system('ls lighning_logs/*') == 0:
                os.system(
                    'for k in $(ls lightning_logs); do rm -r lightning_logs/$k; done')
        valo = {
            'hiddens': n_hidden,
            'batch_size': batch_size,
            'epochs': mlp_pl_cr.epoch,
            'lr_init': lr_init,
            't_ann': t_ann,
            't_max': t_max,
            't_burn': t_burn,
            'lossfun': loss_fun,
            'input_ADN': input_adn,
            'output_ADN': output_adn,
            'skipnet': use_skip_connections
        }
        valo |= calc_metrics(mlp_pl_cr, df, y_columns, train_ids, valid_ids)
        valo |= dict(
            constraint_opt=mlp_pl_cr.model.constraint_loss_model is not None,
            constraint_weight=constraint_weight,
            constraint_x_cols=constraint_x_cols,
            constraint_loss_model=constraint_loss_model
        )
        valas.append(valo)
        print(f'trained for total {valo["epochs"]} epochs')
        print(f'vloss for {n_hidden} hiddens')
        print(valo['va_scr'])
        print(f'tloss for {n_hidden} hiddens')
        print(valo['tr_scr'])
        print('prediction variance test:')
        print(f'mad_x_variance: {valo["mad_x_variance"]}')
        print(f'mad_y_variance: {valo["mad_y_variance"]}')
        mlp_pl_cr.metrics = valo
        js_acc = valo.get('va_js_acc', None)
        if js_acc is None:
            js_acc = 1 - valo.get('va_non_zero_js_div', np.nan)
        mad_acc = valo.get('va_mad_acc', None)
        if mad_acc is None:
            mad_acc = 1 - valo.get('va_non_zero_mae')
        mlp_pl_cr.js_acc = js_acc
        mlp_pl_cr.mad_acc = mad_acc
        if 's3://' in checkpoint_path:
            dump_to(
                mlp_pl_cr,
                bucket_name,
                bucket_dir,
                f'mlp_run_{max_run + j}.pt'
            )
        else:
            joblib.dump(mlp_pl_cr, checkpoint_path + f'/mlp_run_{max_run + j}.pt')
        if return_best_model:
            if best_mdl is None or best_mtr != best_mtr:
                best_mdl = mlp_pl_cr
                best_mtr = valo.get(model_desc_mtr, valo.get('va_non_zero_mae'))
            else:
                desc = (1 - 2 * (model_desc_fun == 'min'))
                mtr = valo.get(model_desc_mtr, valo.get('va_non_zero_mae'))
                if desc * (mtr - best_mtr) > 0:
                    best_mdl = mlp_pl_cr
                    best_mtr = mtr
    pd.DataFrame(
        valas,
        index=[f'run_{k}' for k in range(max_run, max_run + j + 1)]
    ).to_csv(
        checkpoint_path + f'/mtr_mlp_runs_{max_run}-{max_run + j}.csv'
    )
    if return_best_model:
        valas = (best_mdl, valas)
    return valas


def calc_cb_metrics(
        cb_model,
        dfx: pd.DataFrame,
        dfy: pd.DataFrame,
        train_ids: list = None,
        valid_ids: list = None
) -> dict:
    metrics = {
        'va_scr': cb_model.score(
            dfx.loc[valid_ids].fillna(0),
            dfy.loc[valid_ids].fillna(0)
        ),
        'tr_scr': cb_model.score(
            dfx.loc[train_ids].fillna(0),
            dfy.loc[train_ids].fillna(0)
        )
    }
    # pvar_test = predict_variance_test(cb_model, df.loc[valid_ids].fillna(0))
    # metrics['mad_x_variance'] = np.nanmean(pvar_test['feature2prediction_variance'])
    # metrics['mad_y_variance'] = np.nanmean(pvar_test['target2prediction_variance'])
    tr_p = torch.tensor(
        cb_model.predict(dfx.loc[train_ids].fillna(0))
    ).float()
    va_p = torch.tensor(
        cb_model.predict(dfx.loc[valid_ids].fillna(0))
    ).float()
    tr_y = torch.tensor(dfy.loc[train_ids].fillna(0).values).float()
    va_y = torch.tensor(dfy.loc[valid_ids].fillna(0).values).float()
    tr_nozero = np.where(tr_y != 0)[0].tolist()
    va_nozero = np.where(va_y != 0)[0].tolist()
    metrics['va_PNLL'] = PoissonNLLLoss()(
        va_p,
        va_y
    )
    metrics['tr_PNLL'] = PoissonNLLLoss()(
        tr_p,
        tr_y
    )
    metrics['tr_non_zero_mae'] = L1Loss()(tr_p[tr_nozero], tr_y[tr_nozero]).item()
    metrics['va_non_zero_mae'] = L1Loss()(va_p[va_nozero], va_y[va_nozero]).item()
    metrics['tr_non_zero_js_div'] = compute_js_divergence(
        tr_p[tr_nozero], tr_y[tr_nozero], n_bins=96
    ).item()
    metrics['va_non_zero_js_div'] = compute_js_divergence(
        va_p[va_nozero], va_y[va_nozero], n_bins=96
    ).item()
    metrics['tr_mae'] = np.nanmean(np.abs(tr_p - tr_y))
    metrics['va_mae'] = np.nanmean(np.abs(va_p - va_y))
    metrics['tr_mad_acc'] = 1 - metrics['tr_mae'] / tr_y.std().item()
    metrics['va_mad_acc'] = 1 - metrics['va_mae'] / va_y.std().item()
    return metrics


def tune_cp_model_skopt(
        model_class,
        dfx: pd.DataFrame,
        dfy: pd.DataFrame,
        cb_dimensions: list = [
            space.Real(1e-5, 1e-2, name='learning_rate'),
            space.Real(1.2, 4, name='fold_len_multiplier'),
            space.Integer(8, 60, name='bagging_temperature'),
            space.Real(.1, .9, name='reg_lambda'),
            space.Integer(80, 1200, name='leaf_estimation_iterations')
        ],
        cb_defaults: dict = dict(
            iterations=7000,
            metric_period=10,
            thread_count=NCPU,
            bootstrap_type='Bayesian',
            langevin=True,
            posterior_sampling=True,
            objective='RMSE',
            eval_metric="R2"
        ),
        train_idz: list = None,
        valid_idz: list = None,
        checkpoint_path: str = './models/hpo',
        num_steps: int = 60,
        num_ini_points: int = 30,
        train_flag: bool = False,
        return_best_model: bool = False
):
    # make path
    if not train_flag:
        checkpoint_path = os.path.join(checkpoint_path, f'{dt.now().timestamp()}')
    if 's3:' not in checkpoint_path.lower():
        mkpath(checkpoint_path)
        bucket_name = None
        bucket_dir = None
    else:
        bucket_name = checkpoint_path.split('s3://')[1].split('/')[0]
        bucket_dir = '/'.join(checkpoint_path.split('s3://')[1].split('/')[1:])

    if train_idz is None:
        train_idz = dfx.index.values[:0.8 * len(dfx)]
        valid_idz = dfx.index.values[0.8 * len(dfx):]

    # score tracker
    r2_list = {}

    @use_named_args(dimensions=cb_dimensions)
    def tune_cb(**params):
        max_run = get_max_runs(checkpoint_path)
        for param in cb_defaults:
            if param in params:
                logger.warn(f'parameter {param} defined twice, use default!')
            params[param] = cb_defaults[param]
        try:
            if 'catboost' in str(model_class).lower() and valid_idz is not None:
                cb_model = model_class(**params).fit(
                    dfx.fillna(0).loc[train_idz],
                    dfy.fillna(0).loc[train_idz],
                    eval_set=(
                        dfx.fillna(0).loc[valid_idz],
                        dfy.fillna(0).loc[valid_idz]
                    )
                )
            else:
                cb_model = model_class(**params).fit(
                    dfx, dfy
                )
            metrics = {'run': max_run + 1}
            metrics |= params
            metrics |= calc_cb_metrics(cb_model, dfx, dfy, train_idz, valid_idz)
            pd.DataFrame([metrics]).to_csv(
                checkpoint_path + f'/mtr_cb_run_{max_run + 1}.csv'
            )
            # wrapping the model
            cb_model_obj = SciKitRegressor(
                feature_columns=dfx.columns.tolist(),
                target_columns=dfy.columns.tolist(),
                model_class=model_class,
                **params
            )
            cb_model_obj.model = cb_model
            cb_model = cb_model_obj
            del cb_model_obj
            # saving the checkpoint
            if 's3://' in checkpoint_path:
                dump_to(
                    cb_model,
                    bucket_name,
                    bucket_dir,
                    f'cb_run_{max_run + 1}.pkl'
                )
            else:
                joblib.dump(cb_model, checkpoint_path + f'/cb_run_{max_run + 1j}.pkl')
            scr = r2(
                cb_model.predict(dfx.fillna(0).loc[valid_idz]),
                dfy.fillna(0).loc[valid_idz]
            )
            r2_list[f'cb_run_{max_run + 1j}.pkl'] = scr
            err = 1 - scr
            assert np.isfinite(err)
        except Exception as e:  # noqa
            logger.warn(e, exc_info=True)
            logger.warn(f'Unable to run trial {max_run + 1} with parameters {params}')
            err = -900000
        return err

    cb_opt = gp_minimize(
        tune_cb,
        cb_dimensions,
        n_calls=num_steps,
        n_initial_points=num_ini_points,
        initial_point_generator='halton',
        model_queue_size=5
    )

    if not return_best_model:
        return cb_opt
    else:
        if len(r2_list) > 0:
            r2_list = pd.Series(r2_list)
            best_run = r2_list.sort_values(ascending=False).index.values[0]
            try:
                cb_model = get_obj(checkpoint_path + '/' + best_run)
                return cb_opt, cb_model
            except Exception as e:  # noqa
                logger.warn(e.__traceback__)
                logger.warn(e)
                logger.warn(f'Cannot load model from location: {checkpoint_path}/{best_run}')  # noqa
                return cb_opt, None
        else:
            logger.warn('Cannot load model tracks!')  # noqa
            return cb_opt, None
