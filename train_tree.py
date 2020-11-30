import os
import gc
import sys
import hydra
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

from src.load_preprocess import load_and_preprocess_data_index
from src.cv.multilabel import DrugAwareMultilabelStratifiedKFold
from src.torch_model_loop import run_k_fold, run_k_fold_nn
from src.tree.xgb import get_xgboost

os.listdir('../input/lish-moa')

@hydra.main(config_path="config", config_name="config.yaml", strict=False)
def run(cfg: DictConfig) -> None:
    os.chdir(hydra.utils.get_original_cwd())
    log.info(OmegaConf.to_yaml(cfg))
    cfg['device'] = ('cuda' if torch.cuda.is_available() else 'cpu')
    cfg['list_seed'] = [i for i in range(cfg.model.nseed)]
    verbose = 0
    local_path = '../'
    path = f'{local_path}input/lish-moa'
    path_model = f'{local_path}models'
    cfg['path_model'] = path_model
    # print(os.listdir(f'{local_path}../'))
    os.chdir(hydra.utils.get_original_cwd())
    log.info(OmegaConf.to_yaml(cfg))
    cfg['device'] = ('cuda' if torch.cuda.is_available() else 'cpu')
    cfg['list_seed'] = [i for i in range(cfg.model.nseed)]
    verbose = 1
    local_path = '../'
    path = f'{local_path}input/lish-moa'
    path_model = f'{local_path}models'
    cfg['path_model'] = path_model
    # print(os.listdir(f'{local_path}../'))

    ######################################
    # data_load and preprocess
    ######################################

    pretrain_model = False
    data_dict = load_and_preprocess_data_index(cfg, path, pca_append_test=True, variancethreshold_append_test=False, verbose=1)

    CV = DrugAwareMultilabelStratifiedKFold(n_splits=cfg.model.nfolds, shuffle=False, random_state=42)
    ##################################################
    # Train
    ##################################################
    SEED = [0]
    oof = np.zeros((len(data_dict['train']), len(data_dict['target_cols'])))
    predictions = np.zeros((len(data_dict['test']), len(data_dict['target_cols'])))
    for seed in tqdm([0], leave=verbose):
        xgb_params = {
            'booster': 'gbtree',
            'tree_method': 'gpu_hist',
            'min_child_weight': 31.58,
            'learning_rate': 0.05,
            'colsample_bytree': 0.65,
            'gamma': 3.69,
            'max_delta_step': 2.07,
            'max_depth': 10,
            'n_estimators': 10,
            'subsample': 0.86,
            'verbosity': 1,
        }
        return_run_k_fold = get_xgboost(data_dict, cfg, xgb_params, CV, seed=seed,
                                        file_prefix='x1', optimization=False, verbose=0)

        if cfg.model.train_models:
            oof_, predictions_ = return_run_k_fold
            oof += oof_ / len(SEED)
        else:
            predictions_ = return_run_k_fold
        predictions += predictions_ / len(SEED)
        gc.collect()

    train = data_dict['train'].copy()
    test = data_dict['test'].copy()
    target = data_dict['target'].copy()
    feature_cols = data_dict['feature_cols']
    target_cols = data_dict['target_cols']
    train_targets_scored = data_dict['train_targets_scored']
    test_features = data_dict['test_features']

    if not pretrain_model:
        train[target_cols] = oof
    test[target_cols] = predictions

    ##################################################
    # valodation and save
    ##################################################

    if not pretrain_model:
        y_true = train_targets_scored[target_cols].values
        valid_results = train_targets_scored.drop(columns=target_cols).merge(train[target_cols],
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

    res = test[target_cols]
    corner_case = test_features[test_features['cp_type'] == 'ctl_vehicle']
    zeros = np.zeros((corner_case.shape[0], len(target_cols)))
    corner_case[target_cols] = zeros
    corner_case = corner_case[target_cols]
    res = pd.concat([res, corner_case], axis=0)

    res.to_csv('submission.csv')
    log.info(f"res.shape: {res.shape}")
    log.info(f"test[target_cols].shape: {test[target_cols].shape}")

    if not pretrain_model:
        return score
    else:
        return 0


if __name__ == '__main__':
    run()
