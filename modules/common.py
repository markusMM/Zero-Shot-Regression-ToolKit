import os
import json
import torch
import multiprocessing
from configparser import ConfigParser
from modules.log import logger

# smoketest
SAVETY = 2
if (
    '/home/' not in os.popen('echo ~').read() or
    'ec2-user' in os.popen('echo ~').read()
) and not os.environ.get('CI_RUNNER_ID', False):
    NCPU = multiprocessing.cpu_count()
else:
    NCPU = multiprocessing.cpu_count() - SAVETY
if not os.environ.get('CI_RUNNER_ID', False):
    torch.set_num_threads(NCPU)  # torch.set_num_interop_threads(NCPU)
else:
    NCPU = 1

ROOT = os.getcwd()
logger.info(ROOT)

conf_file = os.environ.get('feature_conf_path', 'config/example_features.conf')
conf = ConfigParser()
conf.read(conf_file)

model_a_vars = conf.get("model_a", "features")
model_b_vars = conf.get("model_b", "features")
model_b_kpis = conf.get("model_a", "kpis")
kpis = conf.get("model_a", "kpis")

try:
    assert sum(map(lambda kpi: kpi in kpis, model_b_kpis)) <= 0
    assert sum(map(lambda kpi: kpi in model_b_kpis, kpis)) <= 0
except AssertionError:
    logger.warn('Different KPIs across models is not implemented!')

logger.info('Using [model_a] variant of KPIs!')

model_a_class = os.environ.get(
    'MODEL_A_CLASS',
    'LightningModel'
)
model_a_param = os.environ.get(
    'MODEL_A_PARAM',
    '{}'
)
model_b_class = os.environ.get(
    'CP_MODEL_CLASS',
    'SciKitRegressor'
)
model_b_param = os.environ.get(
    'CP_MODEL_PARAM',
    '{"model": CatBoostRegressor}'
)

ml_pipeline = os.environ.get(
    'ML_PIPELINE_CLASS',
    'SkOptRecPipe'
)

# additional tuning parameters
train_dtypes = os.environ.get('train_dtype_path', None)
if train_dtypes is not None:
    try:
        train_dtypes = json.load(oppen(train_dtypes, 'r+'))
    except FileNotFoundError:
        logger.warning(f'Cannot find data type JSON for training data: {train_dtypes}')
        train_dtypes = None

# administrative parameters
sql_type = os.environ.get("sql_type", 'mysql')
stats_db_host = os.environ.get("stats_db_host")
stats_db_user = os.environ.get("stats_db_user")
stats_db_password = os.environ.get("stats_db_password")
model_bucket = os.environ.get('model_bucket')
