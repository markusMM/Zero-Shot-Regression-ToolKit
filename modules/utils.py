import glob
import os
import sys
import re

import joblib

from modules.data_retrieval import s3_drvr as s3
import numpy as np
import joblib
from modules.data_retrieval import s3_drvr as s3
sys.path.append('..')
CD = '/'.join(__file__.replace('\\', '/').split('/')[:-2])


def glob_file(fpath):
    try:
        fpath = fpath.replace('./', '')
        return (
            glob.glob(os.path.join(
                CD, '**', fpath
            )) + glob.glob(os.path.join(
                CD, fpath
            ))
        )[0]
    except:  # noqa
        return None


def parse_obj_path(
    path,
    suffixes: list = None,
    default_suffix: str = None,
    default_obj_name: str = None
) -> str:
    if default_obj_name is None:
        default_obj_name = 'model'
    if default_suffix is None:
        default_suffix = '.tar.gz'
    if suffixes is None:
        suffixes = ['.pkl', '.tar.gz']
    if suffixes is str:
        suffixes = [suffixes]
    suffix = list(filter(
        lambda suff: re.search(f'{suff}$', path),
        suffixes
    ))
    if len(suffix) <= 0:
        suffix = default_suffix
        obj = default_obj_name
        return os.path.join(path, obj, suffix)
    else:
        return path


def root_names(sarr: list):
    svar = np.array(list(sarr[0]))
    k = len(sarr) - 1
    while len(svar) > 0 and k >= 0:
        if svar[-1] == '_':
            svar = svar[:-1]
        s = np.array(list(sarr[k]))[:len(svar)]
        svar = svar[svar == s]
        k -= 1
    return ''.join(svar) if len(svar) > 0 else None


def get_obj(path):
    if 's3://' in path:
        return s3.load_from(**s3.parse_path(path))
    else:
        return joblib.load(glob_file(path))
