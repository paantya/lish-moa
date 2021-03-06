import os
import gc
import sys
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

def load_and_preprocess_data_index(cfg, path, pca_append_test=False, variancethreshold_append_test=False, verbose=0, testX4=False):
    # data_load
    train_features = pd.read_csv(f'{path}/train_features.csv').set_index('sig_id')
    test_features = pd.read_csv(f'{path}/test_features.csv').set_index('sig_id')

    if testX4:
        print(f"Run X4 test")
        log.info(f"Run X4 test")
        test_features = pd.concat([test_features, test_features, test_features, test_features], axis=0)
        test_features.index = range(test_features.shape[0])
    train_targets_scored = pd.read_csv(f'{path}/train_targets_scored.csv').set_index('sig_id')
    train_targets_nonscored = pd.read_csv(f'{path}/train_targets_nonscored.csv').set_index('sig_id')
    train_drug = pd.read_csv(f'{path}/train_drug.csv').set_index('sig_id')

    train_features = change_type(train_features)
    test_features = change_type(test_features)
    train_targets_scored = change_type(train_targets_scored)
    log.info(f"train_targets_scored.shape: {train_targets_scored.shape}")
    sample_submission = pd.read_csv(f'{path}/sample_submission.csv')
    # sub = pd.read_csv(f'{path}/sample_submission.csv')

    log.info(f"n_comp_genes: {cfg.pca.n_comp_genes}, n_comp_cells: {cfg.pca.n_comp_cells}, total: "
             f"{cfg.pca.n_comp_genes + cfg.pca.n_comp_cells}.")

    GENES = [col for col in train_features.columns if col.startswith('g-')]
    CELLS = [col for col in train_features.columns if col.startswith('c-')]

    train_features_return, test_features_return = \
        quantile_transformer(train_features, test_features, features=GENES + CELLS,
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
        get_pca_transform(train_features, test_features, features=GENES, n_components=cfg.pca.n_comp_genes,
                          flag='g', append_test=pca_append_test)
    train_features = pd.concat((train_features, train_features_return), axis=1)
    test_features = pd.concat((test_features, test_features_return), axis=1)
    del train_features_return, test_features_return
    gc.collect()


    train_features_return, test_features_return = \
        get_pca_transform(train_features, test_features, features=CELLS, n_components=cfg.pca.n_comp_cells,
                          flag='c', append_test=pca_append_test)
    train_features = pd.concat((train_features, train_features_return), axis=1)
    test_features = pd.concat((test_features, test_features_return), axis=1)
    del train_features_return, test_features_return
    gc.collect()
    ##################################################
    # Start: Feature selection
    ##################################################
    train_features_return, test_features_return = \
        split_with_variancethreshold(train_features, test_features,
                                     variance_threshold_for_fs=cfg.variance_threshold.variance_threshold_for_fs,
                                     # categorical=['sig_id', 'cp_type', 'cp_time', 'cp_dose', 'drug_id'],
                                     categorical=['cp_type', 'cp_time', 'cp_dose'],
                                     append_test=variancethreshold_append_test)
    del train_features, test_features
    gc.collect()
    train_features = train_features_return
    test_features = test_features_return

    ##################################################
    # Start: Zero hack target & prepare train test
    ##################################################
    if verbose:
        print(f"Preparation of train & test:")

    train_features = train_features.merge(train_drug, left_index=True, right_index=True)
    train = train_features.merge(train_targets_scored, on='sig_id')
    # train = train[train['cp_type'] != 'ctl_vehicle'].reset_index(drop=True)
    train = train[train['cp_type'] != 'ctl_vehicle']
    # test = test_features[test_features['cp_type'] != 'ctl_vehicle'].reset_index(drop=True)
    test = test_features[test_features['cp_type'] != 'ctl_vehicle']

    target = train[train_targets_scored.columns]
    train = train.drop('cp_type', axis=1)
    test = test.drop('cp_type', axis=1)

    target_cols = target.columns.values.tolist()

    log.debug(f"Preparation of train & test.\n"
              f"train.shape: {train.shape}\n"
              f"test.shape: {test.shape}\n"
              f"{'_' * 80}\n"
              )

    ##################################################
    # cv folds
    ##################################################
    # folds = train.copy()
    # mskf = MultilabelStratifiedKFold(n_splits=cfg.mode.nfolds, random_state=cfg['list_seed'][0])
    #
    # for f, (t_idx, v_idx) in enumerate(mskf.split(X=train, y=target)):
    #     folds.loc[v_idx, 'kfold'] = int(f)
    #
    # folds['kfold'] = folds['kfold'].astype(int)

    # log.debug(f"train.shape: {train.shape}"
    #           f"folds.shape: {folds.shape}"
    #           f"test.shape: {test.shape}"
    #           f"target.shape: {target.shape}"
    #           )

    gc.collect()
    ##################################################
    # Preprocessing feature_cols
    ##################################################
    feature_cols = [c for c in preprocess_data(train, cfg.mode.patch1).columns if c not in target_cols]
    feature_cols = [c for c in feature_cols if c not in ['kfold', 'sig_id', 'drug_id']]
    num_features = len(feature_cols)
    num_targets = len(target_cols)

    ##################################################
    # END PREPROCESS
    ##################################################
    train_features
    data_dict = {
        'train': preprocess_data(train),
        'target': target,
        'test': preprocess_data(test),
        'feature_cols': feature_cols,
        'target_cols': target_cols,
        'train_features': train_features,
        'train_targets_scored': train_targets_scored,
        'train_targets_nonscored': train_targets_nonscored,
        'test_features': test_features,
    }
    # base_model_def(data_dict, params, cv=cv, optimization=False, verbose=0):
    return data_dict

def load_and_preprocess_data(cfg, path, pca_append_test=False, variancethreshold_append_test=False, verbose=0):
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
        quantile_transformer(train_features, test_features, features=GENES + CELLS,
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
                          flag='g', append_test=pca_append_test)
    train_features = pd.concat((train_features, train_features_return), axis=1)
    test_features = pd.concat((test_features, test_features_return), axis=1)
    del train_features_return, test_features_return
    gc.collect()

    train_features_return, test_features_return = \
        get_pca_transform(train_features, test_features, features=CELLS, n_components=cfg.model.n_comp_cells,
                          flag='c', append_test=pca_append_test)
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
                                     append_test=variancethreshold_append_test)
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
    # cv folds
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

    ##################################################
    # END PREPROCESS
    ##################################################

    data_dict = {
        'train': preprocess_data(train),
        'target': target,
        'test': preprocess_data(test),
        'feature_cols': feature_cols,
        'target_cols': target_cols,
        'train_targets_scored': train_targets_scored,
        'test_features': test_features,
    }
    # base_model_def(data_dict, params, cv=cv, optimization=False, verbose=0):
    return data_dict

# Press the green button in the gutter to run the script.
if __name__ == '__main__':
    print('PyCharm')

# See PyCharm help at https://www.jetbrains.com/help/pycharm/
