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

from src.torch_model_loop import run_k_fold
from src.data.process_data import set_seed, preprocess_data, change_type, from_yml, \
    quantile_transformer, get_pca_transform, split_with_variancethreshold

os.listdir('../input/lish-moa')

@hydra.main(config_path="config", config_name="config.yaml", strict=False)
def run(cfg: DictConfig) -> None:
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

    # data_load
    train_features = pd.read_csv(f'{path}/train_features.csv')
    test_features = pd.read_csv(f'{path}/test_features.csv')
    train_targets_scored = pd.read_csv(f'{path}/train_targets_scored.csv')
    train_targets_nonscored = pd.read_csv(f'{path}/train_targets_nonscored.csv')

    train_features = change_type(train_features)
    test_features = change_type(test_features)
    train_targets_scored = change_type(train_targets_scored)
    log.info(f"train_targets_scored.shape: {train_targets_scored.shape}")
    sample_submission = pd.read_csv(f'{path}/sample_submission.csv')
    # sub = pd.read_csv(f'{path}/sample_submission.csv')

    log.info(f"n_comp_genes: {cfg.model.n_comp_genes}, n_comp_cells: {cfg.model.n_comp_cells}, total: "
          f"{cfg.model.n_comp_genes + cfg.model.n_comp_cells}.")

    GENES = [col for col in train_features.columns if col.startswith('g-')]
    CELLS = [col for col in train_features.columns if col.startswith('c-')]

    train_features_return, test_features_return = \
        quantile_transformer(train_features, test_features, features=GENES+CELLS,
                             n_quantiles=cfg.quantile_transformer.n_quantiles,
                             output_distribution=cfg.quantile_transformer.output_distribution)
    del train_features, test_features
    gc.collect()
    train_features = train_features_return
    test_features = test_features_return
    log.info(f"End prearation data transform.\n"
             f"train_features.shape: {train_features.shape}\n"
             f"test_features.shape: {test_features.shape}\n"
             f"{'_' * 80}\n")

    ##################################################
    # PCA
    ##################################################

    train_features_return, test_features_return = \
        get_pca_transform(train_features, test_features, features=GENES, n_components=cfg.model.n_comp_genes,
                          flag='GENES', test_append=False)
    train_features = pd.concat((train_features, train_features_return), axis=1)
    test_features = pd.concat((test_features, test_features_return), axis=1)
    del train_features_return, test_features_return
    gc.collect()

    train_features_return, test_features_return = \
        get_pca_transform(train_features, test_features, features=CELLS, n_components=cfg.model.n_comp_cells,
                          flag='CELLS', test_append=False)
    train_features = pd.concat((train_features, train_features_return), axis=1)
    test_features = pd.concat((test_features, test_features_return), axis=1)
    del train_features_return, test_features_return
    gc.collect()
    ##################################################
    # Start: Feature selection
    ##################################################
    train_features_return, test_features_return = \
        split_with_variancethreshold(train_features, test_features,
                                     variance_threshold_for_fs=cfg.model.variance_threshold_for_fs,
                                     categorical=['sig_id', 'cp_type', 'cp_time', 'cp_dose'],
                                     test_append=False)
    del train_features, test_features
    gc.collect()
    train_features = train_features_return
    test_features = test_features_return

    ##################################################
    # Start: Zero hack target & prepare train test
    ##################################################
    if verbose:
        print(f"Preparation of train & test:")

    train = train_features.merge(train_targets_scored, on='sig_id')
    train = train[train['cp_type'] != 'ctl_vehicle'].reset_index(drop=True)
    test = test_features[test_features['cp_type'] != 'ctl_vehicle'].reset_index(drop=True)

    target = train[train_targets_scored.columns]
    train = train.drop('cp_type', axis=1)
    test = test.drop('cp_type', axis=1)

    target_cols = target.drop('sig_id', axis=1).columns.values.tolist()

    log.debug(f"Preparation of train & test.\n"
              f"train.shape: {train.shape}\n"
              f"test.shape: {test.shape}\n"
              f"{'_' * 80}\n"
              )

    ##################################################
    # CV folds
    ##################################################
    folds = train.copy()
    mskf = MultilabelStratifiedKFold(n_splits=cfg.model.nfolds, random_state=cfg['list_seed'][0])

    for f, (t_idx, v_idx) in enumerate(mskf.split(X=train, y=target)):
        folds.loc[v_idx, 'kfold'] = int(f)

    folds['kfold'] = folds['kfold'].astype(int)

    log.debug(f"train.shape: {train.shape}"
              f"folds.shape: {folds.shape}"
              f"test.shape: {test.shape}"
              f"target.shape: {target.shape}"
              )

    gc.collect()
    ##################################################
    # Preprocessing feature_cols
    ##################################################
    feature_cols = [c for c in preprocess_data(folds, cfg.model.patch1).columns if c not in target_cols]
    feature_cols = [c for c in feature_cols if c not in ['kfold', 'sig_id']]
    num_features = len(feature_cols)
    num_targets = len(target_cols)

    # Averaging on multiple SEEDS

    #     print(f"target.columns: {target.columns}")

    ##################################################
    # Train
    ##################################################
    SEED = cfg['list_seed']
    oof = np.zeros((len(train), len(target_cols)))
    predictions = np.zeros((len(test), len(target_cols)))

    for seed in tqdm(SEED, leave=verbose):
        return_run_k_fold = run_k_fold(cfg.model.nfolds, seed, cfg, folds, train, test, feature_cols, target_cols,
                                       num_features, num_targets, target, verbose)
        if cfg.model.train_models:
            oof_, predictions_ = return_run_k_fold
            oof += oof_ / len(SEED)
        else:
            predictions_ = return_run_k_fold
        predictions += predictions_ / len(SEED)
        gc.collect()

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
            score += score_ / num_targets

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
