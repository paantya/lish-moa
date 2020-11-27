import os
import gc
import torch
import random
import inspect
import logging
import numpy as np
import pandas as pd

from tqdm.auto import tqdm as tqdm
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import QuantileTransformer
from sklearn.feature_selection import VarianceThreshold

log = logging.getLogger(__name__)

import yaml
import io
from omegaconf import DictConfig, OmegaConf

def from_yml(file='data.yaml',  verbose=1):
    """

    :param file:
    :return:
    """
    log = logging.getLogger(f"{__name__}.{inspect.currentframe().f_code.co_name}")
    # Read YAML file
    log.info(f"Read .yaml from {file}")
    if os.path.exists(file):
        with io.open(file, 'r') as stream:
            data_loaded = None
            try:
                data_loaded = yaml.safe_load(stream)
            except yaml.YAMLError as exc:
                log.info(f"yaml.YAMLError: {exc}")
        return data_loaded
    else:
        if verbose:
            log.info(f"File not exists: file=`{file}`")
        else:
            log.debug(f"File not exists: file=`{file}`")
        return {}

def set_seed(seed, precision: int = 10) -> None:
    random.seed(seed)
    np.random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.set_printoptions(precision=precision)
    torch.backends.cudnn.benchmark = False  # not stable
    torch.backends.cudnn.deterministic = True  # not stable


def change_type(data):
    for k, v in data.dtypes.items():
        if v == 'float64':
            data[k] = data[k].astype('float32')
        if v == 'int64':
            data[k] = data[k].astype('int8')
    return data

# Preprocessing
def preprocess_data(data, patch1=False):
    log = logging.getLogger(f"{__name__}.{inspect.currentframe().f_code.co_name}")
    if patch1:
        cp_dose_int = data['cp_dose'].map({"D1": 1., "D2": 0.})
        cp_time_int = data['cp_time'].map({24: 1., 48: 2., 72: 3.})
    data = pd.get_dummies(data, columns=['cp_time', 'cp_dose'])
    if 'cp_type' in data.columns:
        data = pd.get_dummies(data, columns=['cp_type'])
    if patch1:
        data['cp_dose_int'] = cp_dose_int
        data['cp_time_int'] = cp_time_int
    log.debug(f"data.shape: {data.shape}")
    return data


def quantile_transformer(train_features, test_features, features, n_quantiles=100, output_distribution='normal'):
    log = logging.getLogger(f"{__name__}.{inspect.currentframe().f_code.co_name}")
    log.info("Start.one_experiment")
    train_features = train_features.copy()
    test_features = test_features.copy()

    ##################################################
    # RankGauss - transform to Gauss
    ##################################################

    log.debug(f"Prearation data transform.\ntrain_features.shape: {train_features.shape}")
    for col in tqdm(features, 'QuantileTransformer', leave=False):
        # kurt = max(kurtosis(train_features[col]), kurtosis(test_features[col]))
        # QuantileTransformer_n_quantiles = n_quantile_for_kurt(kurt, calc_QT_par_kurt(QT_n_quantile_min, QT_n_quantile_max))
        # transformer = QuantileTransformer(n_quantiles=QuantileTransformer_n_quantiles,random_state=0, output_distribution="normal")

        transformer = QuantileTransformer(n_quantiles=n_quantiles, random_state=0,
                                          output_distribution=output_distribution)  # from optimal commit 9
        vec_len = len(train_features[col].values)
        vec_len_test = len(test_features[col].values)
        raw_vec = train_features[col].values.reshape(vec_len, 1)
        transformer.fit(raw_vec)

        train_features[col] = transformer.transform(raw_vec).reshape(1, vec_len)[0]
        test_features[col] = \
            transformer.transform(test_features[col].values.reshape(vec_len_test, 1)).reshape(1, vec_len_test)[0]

    gc.collect()
    return train_features, test_features


def get_pca_transform(train_features, test_features, features, n_components, flag='', random_state=42, test_append=False):
    train_features = train_features.copy()
    test_features = test_features.copy()
    log = logging.getLogger(f"{__name__}.{inspect.currentframe().f_code.co_name}")
    log.info(f"Start PCA {flag} :len({flag}): {len(features)}")
    # PCA GENES
    if test_append:
        data = pd.concat([pd.DataFrame(train_features[features]), pd.DataFrame(test_features[features])])
    else:
        data = train_features[features]
    data2 = (PCA(n_components=n_components, random_state=random_state).fit_transform(data[features]))
    train2 = data2[:train_features.shape[0]]
    test2 = data2[-test_features.shape[0]:]

    train2 = pd.DataFrame(train2, columns=[f'pca_G-{i}' for i in range(n_components)])
    test2 = pd.DataFrame(test2, columns=[f'pca_G-{i}' for i in range(n_components)])

    log.debug(f"End PCA {flag}.\n"
              f"train2.shape: {train2.shape}\n"
              f"test2.shape: {test2.shape}\n"
              f"{'_' * 80}\n"
              )

    gc.collect()
    return train2, test2


def split_with_variancethreshold(train_features, test_features, variance_threshold_for_fs, categorical,
                                 test_append=False):
    log = logging.getLogger(f"{__name__}.{inspect.currentframe().f_code.co_name}")
    log.info(f"Start feature_selection.VarianceThreshold")
    train_features = train_features.copy()
    test_features = test_features.copy()

    if test_append:
        data = train_features.append(test_features)
    else:
        data = train_features

    log.info(f" data.shape (data = {'concat(train_features + test_features)' if test_append else 'train_features'}"
             f"): {data.shape}")

    var_thresh = VarianceThreshold(variance_threshold_for_fs)
    var_thresh.fit(data.iloc[:, 4:])
    train_features_transformed = var_thresh.transform(train_features.iloc[:, 4:])
    test_features_transformed = var_thresh.transform(test_features.iloc[:, 4:])

    train_features = pd.DataFrame(train_features[categorical].values.reshape(-1, len(categorical)), columns=categorical)
    train_features = pd.concat([train_features, pd.DataFrame(train_features_transformed)], axis=1)

    test_features = pd.DataFrame(test_features[categorical].values.reshape(-1, len(categorical)), columns=categorical)
    test_features = pd.concat([test_features, pd.DataFrame(test_features_transformed)], axis=1)

    log.debug(f"End feature_selection.VarianceThreshold.\n"
              f"train_features.shape: {train_features.shape}\n"
              f"test_features.shape: {test_features.shape}\n"
              f"{'_' * 80}\n"
              )

    gc.collect()
    return train_features, test_features
