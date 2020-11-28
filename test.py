# %% [code]
import os
import gc
import sys
import torch
import logging
import warnings
import numpy as np
import pandas as pd

from tqdm.auto import tqdm as tqdm
from sklearn.metrics import log_loss

sys.path.append('../input/omegaconf')
from omegaconf.omegaconf import DictConfig, OmegaConf

log = logging.getLogger(__name__)
warnings.filterwarnings('ignore')
pd.set_option('max_columns', 2000)

##########################

sys.path.append('../input/iterativestratification')
from iterstrat.ml_stratifiers import MultilabelStratifiedKFold

sys.path.append('../input/src-code0')
sys.path.append('../input/models0')

from src.load_preprocess import load_and_preprocess_data
from src.torch_model_loop import run_k_fold, run_k_fold_nn
from src.data.process_data import set_seed, preprocess_data, from_yml, \
    quantile_transformer, get_pca_transform, split_with_variancethreshold

os.listdir('../input/lish-moa')


def change_type(data):
    for k, v in data.dtypes.items():
        if v == 'float64':
            data[k] = data[k].astype('float32')
        if v == 'int64':
            data[k] = data[k].astype('int8')
    return data


# @hydra.main(config_path="config", config_name="config.yaml", strict=False)
def run():
    on_kaggle = False  # change me True if you use kaggle
    cfg = OmegaConf.load(f"{'../input/src-code0' if on_kaggle else './'}/src/test.yaml")
    # os.chdir(utils.get_original_cwd())
    #     log.info(OmegaConf.to_yaml(cfg))
    cfg['device'] = 'cpu'
    print(cfg['device'])
    cfg['list_seed'] = [i for i in range(cfg.model.nseed)]
    verbose = 0
    local_path = '../'
    path = f'../input/lish-moa'
    path_model = f"{'/kaggle/input/models0' if on_kaggle else '../models'}"
    cfg['path_model'] = path_model
    ######################################
    # data_load and preprocess
    ######################################
    data_dict = load_and_preprocess_data(cfg, path, test_append=False, verbose=1)

    ######################################
    # CV
    ######################################
    CV = MultilabelStratifiedKFold(n_splits=cfg.model.nfolds, random_state=42)

    ##################################################
    # Train
    ##################################################
    oof = np.zeros((len(data_dict['train']), len(data_dict['target_cols'])))
    predictions = np.zeros((len(data_dict['test']), len(data_dict['target_cols'])))
    for seed in tqdm(cfg['list_seed'], leave=verbose):
        return_run_k_fold = run_k_fold_nn(data_dict, cfg, cv=CV, seed=seed, file_prefix='m1', verbose=verbose)
        if cfg.model.train_models:
            oof_, predictions_ = return_run_k_fold
            oof += oof_ / cfg.model.nseed
        else:
            predictions_ = return_run_k_fold
        predictions += predictions_ / cfg.model.nseed
        gc.collect()



    train = data_dict['train'].copy()
    test = data_dict['test'].copy()
    target = data_dict['target'].copy()
    feature_cols = data_dict['feature_cols']
    target_cols = data_dict['target_cols']
    train_targets_scored = data_dict['train_targets_scored']
    test_features = data_dict['test_features']

    if cfg.model.train_models:
        train[target_cols] = oof
    test[target_cols] = predictions

    ##################################################
    # valodation and save
    ##################################################

    if cfg.model.train_models:
        y_true = train_targets_scored[target_cols].values
        valid_results = train_targets_scored.drop(columns=target_cols).merge(train[['sig_id'] + target_cols],
                                                                         on='sig_id', how='left').fillna(0)
        y_pred = valid_results[target_cols].values

        score = 0
        for i in range(len(target_cols)):
            score_ = log_loss(y_true[:, i], y_pred[:, i])
            score += score_ / len(target_cols)

        print(f"CV log_loss: {score}")
        log.info(f"CV log_loss: {score}")
        log.info(f"y_true.shape: {y_true.shape}")
        log.info(f"y_pred.shape: {y_pred.shape}")

    # sub = sample_submission.drop(columns=target_cols).merge(test[['sig_id'] + target_cols], on='sig_id',
    #                                                         how='left').fillna(0)
    # sub.to_csv('submission.csv', index=False)
    # log.info(f"sub.shape: {sub.shape}")

    res = test[['sig_id'] + target_cols]
    corner_case = test_features[test_features['cp_type'] == 'ctl_vehicle']
    zeros = np.zeros((corner_case.shape[0], len(target_cols)))
    corner_case[target_cols] = zeros
    corner_case = corner_case[['sig_id'] + target_cols]
    res = pd.concat([res, corner_case], axis=0)

    res.to_csv('submission.csv', index=False)

    if cfg.model.train_models:
        return score
    else:
        return 0


if __name__ == '__main__':
    run()