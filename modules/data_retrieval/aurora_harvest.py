import glob
import pandas as pd
import sqlalchemy as sa
import subprocess
import sys
import dask.dataframe as dd
from modules.common import (
    stats_db_host, stats_db_user, stats_db_password, sql_type
)


def pip_install(package):
    subprocess.check_call([sys.executable, "-m", "pip", "install", package])


def get_lazy_aurora_dd(query, id_name="id", **kwargs):
    if "select" not in query:
        query = load_query_file(query, **kwargs)
    constr = (
        f"{sql_type}://{stats_db_user}:{stats_db_password}"
        f"@{stats_db_host}"
        ":3306"
    )
    return dd.read_sql_query(sa.text(query), constr, id_name)


def get_aurora_query(query, condition: str = None, **kwargs):
    if "select" not in query.lower():
        query = load_query_file(query, **kwargs)
    if condition is not None:
        query = query.replace(";", "") + " " + condition
    con = sa.create_engine(
        f"{sql_type}://{stats_db_user}:{stats_db_password}"
        f"@{stats_db_host}"
        ":3306",
        connect_args={'connect_timeout': 0.1}
    )
    return pd.read_sql(sa.text(query), con.connect())


def load_query_file(path: str, **kwargs) -> str:
    sql_file = glob.glob(path)
    assert len(sql_file)
    with open(sql_file[0], "r") as f:
        query = "".join(f.readlines()).format(**kwargs)
    return query
