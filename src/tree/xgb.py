from datetime import datetime
import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
from tqdm.auto import tqdm as tqdm

from xgboost import XGBClassifier, XGBRegressor
import xgboost as xgb
from sklearn.model_selection import KFold
from category_encoders import CountEncoder
from sklearn.pipeline import Pipeline
from sklearn.metrics import log_loss

import matplotlib.pyplot as plt

from sklearn.multioutput import MultiOutputClassifier
import gc
import sys
sys.path.append('../../../input/iterativestratification')
from iterstrat.ml_stratifiers import MultilabelStratifiedKFold

import os
import warnings
warnings.filterwarnings('ignore')

def metric(y_true, y_pred):
    metrics = []
    metrics.append(log_loss(y_true, y_pred.astype(float), labels=[0,1]))
    return np.mean(metrics)


def get_xgboost_cv(train, targets, test, sub, NFOLDS=7):
    train_score = targets


    train = train.iloc[:, 1:]
    test = test.iloc[:, 1:]
    train_score = targets.iloc[:, 1:]
    sample = sub

    cols = train_score.columns
    submission = sample.copy()
    submission.loc[:, train_score.columns] = 0
    # test_preds = np.zeros((test.shape[0], train_score.shape[1]))
    oof_loss = 0

    for c, column in enumerate(cols, 1):
        y = train_score[column]
        total_loss = 0

        # cv = KFold(n_splits=NFOLDS, shuffle=True).split(train)
        CV = MultilabelStratifiedKFold(n_splits=NFOLDS, random_state=42).split(X=train, y=train_score)

        # print(f"train.shape: {train.shape}")
        # print(f"train_score.shape: {train_score.shape}")
        # print(f"y.shape: {y.shape}")


        # mode = xgb.cv(
        #     params=params,
        #     dtrain=trainDM,
        #     folds=cv,
        #     # num_boost_round=500,
        #     early_stopping_rounds=10,
        # )
        # mode.predict(testDM)

        for fn, (trn_idx, val_idx) in enumerate(CV):
            print('Fold: ', fn + 1)
            X_train, X_val = train.iloc[trn_idx], train.iloc[val_idx]
            y_train, y_val = y.iloc[trn_idx], y.iloc[val_idx]

            dtrain = xgb.DMatrix(X_train.values, label=y_train.values, feature_names=train.columns.values)
            dtest = xgb.DMatrix(X_val.values, label=y_val.values, feature_names=train.columns.values)

            params = {
                'booster': 'gbtree',
                'tree_method': 'gpu_hist',
                'min_child_weight': 31.58,
                'learning_rate': 0.05,
                'colsample_bytree': 0.65,
                # 'gamma': 3.65,
                'max_delta_step': 2.07,
                'max_depth': 10,
                'subsample': 0.86,
                'verbosity': 1,

            }
            model = xgb.train(
                params=params,
                dtrain=dtrain,
                num_boost_round=166,
                # evals=[(dtrain, 'train'), (dtest, 'test')],
                verbose_eval=5,
            )
            pred = model.predict(dtest)

            loss = metric(y_val, pred)
            total_loss += loss
            print(f"loc loss: {loss}")
            predictions = model.predict(xgb.DMatrix(test.values, feature_names=train.columns.values))
            # predictions = [n if n>0 else 0 for n in predictions]
            submission[column] += predictions / NFOLDS

            # for score in ['weight', 'gain', 'cover', 'total_gain', 'total_cover']:
            #     importance = mode.get_score(importance_type=score).items()
            #     if len(importance) > 0:
            #         print(f"importance ({score}): {importance}")

        submission[column] = submission[column]/NFOLDS
        oof_loss += total_loss / NFOLDS
        print("Model " + str(c) + ": Loss =" + str(total_loss / NFOLDS))
    submission.loc[test['cp_type'] == 1, train_score.columns] = 0
    return submission


def get_xgboost_fe(train, targets, test, sub, xgb_params, importance_type='weight', NFOLDS=7, verbosity=0):
    """

    :param train:
    :param targets:
    :param test:
    :param sub:
    :param xgb_params:
    :param importance_type: (def.: 'weight') also choice ['weight', 'gain', 'cover', 'total_gain', 'total_cover']
    :param NFOLDS:
    :param verbosity:
    :return:
    """
    train_score = targets


    train = train.iloc[:, 1:]
    test = test.iloc[:, 1:]
    train_score = targets.iloc[:, 1:]
    sample = sub

    cols = train_score.columns
    submission = sample.copy()
    submission.loc[:, train_score.columns] = 0
    # test_preds = np.zeros((test.shape[0], train_score.shape[1]))
    oof_loss = 0

    start_time = datetime.now()

    fe_dict = {}
    for column in train.columns.values:
        fe_dict[column] = 0.0
    for c, column in enumerate(tqdm(cols, 'models_one_cols'), 1):
        y = train_score[column]
        total_loss = 0

        # cv = KFold(n_splits=NFOLDS, shuffle=True).split(train)
        CV = MultilabelStratifiedKFold(n_splits=NFOLDS, random_state=42).split(X=train, y=targets)

        start_time_loc = datetime.now()
        for fn, (trn_idx, val_idx) in enumerate(CV):
            if verbosity > 1:
                print('\rFold: ', fn + 1, end='')
            X_train, X_val = train.iloc[trn_idx], train.iloc[val_idx]
            y_train, y_val = y.iloc[trn_idx], y.iloc[val_idx]

            model = XGBRegressor(
                **xgb_params
            )

            model.fit(X_train, y_train, )

            importance = model.get_booster().get_score(importance_type=importance_type).items()
            if len(importance) < 1:
                if verbosity:
                    print(f"[column {c} ({column}), CV {fn}] importance len < 1")
            else:
                for k, v in importance:
                    # print(f"{k}: {v}")
                    fe_dict[k] += v/len(cols)
            pred = model.predict(X_val)
            # pred = [n if n>0 else 0 for n in pred]

            loss = metric(y_val, pred)
            total_loss += loss
            predictions = model.predict(test)
            # predictions = [n if n>0 else 0 for n in predictions]
            submission[column] += predictions / NFOLDS

        stop_time_loc = datetime.now()
        submission[column] = submission[column]/NFOLDS
        oof_loss += total_loss / NFOLDS

        if verbosity > 1:
            print(f"\r[{stop_time_loc - start_time_loc}] Model " + str(c) + ": Loss =" + str(total_loss / NFOLDS))

    stop_time = datetime.now()

    if verbosity:
        print(f"[{stop_time - start_time}] oof_loss/len(cols): {oof_loss/len(cols)}")
    # submission.loc[test['cp_type'] == 1, train_score.columns] = 0
    return {k: v for k, v in sorted(fe_dict.items(), key=lambda kv: kv[1], reverse=True)}




def get_xgboost(data_dict, hparams, xgb_params, cv, seed=42, file_prefix='m1', optimization=False, verbose=0):
    # xgb_params, NFOLDS=7, optimization=False, verbosity=0):

    train_features = data_dict['train_features'].copy()
    train_targets_scored = data_dict['train_targets_scored'].copy()
    train = data_dict['train'].copy()
    test = data_dict['test'].copy()
    target = data_dict['target'].copy()
    feature_cols = data_dict['feature_cols']
    target_cols = data_dict['target_cols']


    oof = np.zeros((len(data_dict['train']), len(data_dict['target_cols'])))
    predictions = np.zeros((len(data_dict['test']), len(data_dict['target_cols'])))

    train_score = target[target_cols]
    cols = target_cols
    submission = np.zeros((len(data_dict['test']), len(data_dict['target_cols'])))
    # test_preds = np.zeros((test.shape[0], train_score.shape[1]))
    oof_loss = 0

    start_time = datetime.now()
    for c, column in enumerate(tqdm(target_cols, 'models_one_cols', leave=False)):
        y = train_score[column]
        total_loss = 0

        start_time_loc = datetime.now()
        oof_ = np.zeros(len(train))
        for fold, (trn_idx, val_idx) in enumerate(tqdm(cv.split(X=train[feature_cols+['drug_id']], y=train[target_cols]),
                                                       f'run {hparams.model.nfolds} folds',
                                                       total=hparams.model.nfolds,
                                                       leave=False)):

            X_train, y_train, = train[feature_cols].iloc[trn_idx].values, y.iloc[trn_idx].values
            X_valid, y_valid = train[feature_cols].iloc[val_idx].values, y.iloc[val_idx].values


            model = XGBRegressor(
                **xgb_params
            )

            model.fit(X_train, y_train)
            pred = model.predict(X_valid)

            oof_[val_idx] = pred
            # pred = [n if n>0 else 0 for n in pred]

            loss = metric(y_valid, pred)
            total_loss += loss
            predictions = model.predict(test[feature_cols].values)
            # predictions = [n if n>0 else 0 for n in predictions]
            submission[:, c] += predictions / hparams.model.nfolds

        # if not pretrain_model:
        oof[:, c] += oof_

        stop_time_loc = datetime.now()
        oof_loss += total_loss / hparams.model.nfolds
        if verbose > 1:
            print(f"\r[{stop_time_loc - start_time_loc}] Model " + str(c) + ": Loss =" + str(total_loss / hparams.model.nfolds))
    predictions = submission
    stop_time = datetime.now()

    if verbose:
        print(f"[{stop_time - start_time}] oof_loss/len(target_cols): {oof_loss/len(target_cols)}")
    # submission.loc[test['cp_type'] == 1, train_score.columns] = 0

    gc.collect()
    if optimization:
        return oof_loss/len(target_cols)
    else:
        if hparams.model.train_models:
            return oof, predictions
        else:
            return predictions


def get_xgboost_old(train, targets, test, NFOLDS=7):
    # drop id col
    X = train.iloc[:, 1:].to_numpy()
    X_test = test.iloc[:, 1:].to_numpy()
    y = targets.iloc[:, 1:].to_numpy()

    classifier = MultiOutputClassifier(XGBClassifier(tree_method='gpu_hist'))
    clf = Pipeline([
        ('encode', CountEncoder(cols=[0, 2])),
        ('classify', classifier)
                    ])

    params = {'classify__estimator__colsample_bytree': 0.6522,
              'classify__estimator__gamma': 3.6975,
              'classify__estimator__learning_rate': 0.0503,
              'classify__estimator__max_delta_step': 2.0706,
              'classify__estimator__max_depth': 10,
              'classify__estimator__min_child_weight': 31.5800,
              'classify__estimator__n_estimators': 166,
              'classify__estimator__subsample': 0.8639
              }
    _ = clf.set_params(**params)

    oof_preds = np.zeros(y.shape)
    test_preds = np.zeros((test.shape[0], y.shape[1]))
    oof_losses = []
    kf = KFold(n_splits=NFOLDS)
    for fn, (trn_idx, val_idx) in enumerate(kf.split(X, y)):
        print('Starting fold: ', fn)
        X_train, X_val = X[trn_idx], X[val_idx]
        y_train, y_val = y[trn_idx], y[val_idx]

        # drop where cp_type==ctl_vehicle (baseline)
        ctl_mask = X_train[:, 0] == 'ctl_vehicle'
        X_train = X_train[~ctl_mask, :]
        y_train = y_train[~ctl_mask]

        clf.fit(X_train, y_train)
        val_preds = clf.predict_proba(X_val)  # list of preds per class
        val_preds = np.array(val_preds)[:, :, 1].T  # take the positive class
        oof_preds[val_idx] = val_preds

        # .named_steps['classifier'].feature_importances_\

        # clf.named_steps['classify'].get_score(importance_type='gain')

        loss = log_loss(np.ravel(y_val), np.ravel(val_preds))
        oof_losses.append(loss)
        preds = clf.predict_proba(X_test)
        preds = np.array(preds)[:, :, 1].T  # take the positive class
        test_preds += preds / NFOLDS

    print(oof_losses)
    print('Mean OOF loss across folds', np.mean(oof_losses))
    print('STD OOF loss across folds', np.std(oof_losses))

    # set control train preds to 0
    control_mask = train['cp_type'] == 'ctl_vehicle'
    oof_preds[control_mask] = 0

    print('OOF log loss: ', log_loss(np.ravel(y), np.ravel(oof_preds)))

    # set control test preds to 0
    control_mask = test['cp_type'] == 'ctl_vehicle'

    test_preds[control_mask] = 0
    return test_preds


if __name__ == '__main__':
    import xgboost as xgb
    import sklearn
    print(f"xgb.__version__: {xgb.__version__}")
    print(f"sklearn.__version__: {sklearn.__version__}")
    from src.data.process_data import preprocess_data
    DATA_DIR = '../../../input/lish-moa/'
    train = pd.read_csv(DATA_DIR + 'train_features.csv')
    targets = pd.read_csv(DATA_DIR + 'train_targets_scored.csv')

    test = pd.read_csv(DATA_DIR + 'test_features.csv')
    sub = pd.read_csv(DATA_DIR + 'sample_submission.csv')

    #  0   sig_id   object
    #  1   cp_type  object
    #  2   cp_time  int64
    #  3   cp_dose  object

    # train = train.drop('cp_type', axis=1)
    # test = test.drop('cp_type', axis=1)
    # train = train.drop('cp_dose', axis=1)
    # test = test.drop('cp_dose', axis=1)

    train = preprocess_data(train)
    test = preprocess_data(test)

    # xgb_params = {
    #     'booster': 'gbtree',
    #     'tree_method': 'gpu_hist',
    #     'min_child_weight': 31.58,
    #     'learning_rate': 0.05,
    #     'colsample_bytree': 0.65,
    #     'gamma': 3.69,
    #     'max_delta_step': 2.07,
    #     'max_depth': 10,
    #     'n_estimators': 166,
    #     'subsample': 0.86,
    #     'verbosity': 1,
    # }
    # get_xgboost1(train=train, targets=targets, test=test, sub=sub, xgb_params=xgb_params, NFOLDS=2, verbosity=2)

    xgb_params = {
        'booster': 'gbtree',
        'tree_method': 'gpu_hist',
        'learning_rate': 0.05,
        'colsample_bytree': 0.65,
        'max_depth': 10,
        'n_estimators': 10,
        'subsample': 0.86,
        'verbosity': 1,
    }
    fe = get_xgboost_fe(train=train, targets=targets, test=test, sub=sub, xgb_params=xgb_params,
                        importance_type='weight', NFOLDS=2, verbosity=1)
    for k, v in fe.items():
        print(f"{k}: {v}.")

        # total_gain