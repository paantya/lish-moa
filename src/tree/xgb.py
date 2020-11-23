import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

from xgboost import XGBClassifier, XGBRegressor
from sklearn.model_selection import KFold
from category_encoders import CountEncoder
from sklearn.pipeline import Pipeline
from sklearn.metrics import log_loss

import matplotlib.pyplot as plt

from sklearn.multioutput import MultiOutputClassifier

import os
import warnings
warnings.filterwarnings('ignore')

def metric(y_true, y_pred):
    metrics = []
    metrics.append(log_loss(y_true, y_pred.astype(float), labels=[0,1]))
    return np.mean(metrics)

def get_xgboost1(train, targets, test, sub, NFOLDS=7):
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

        for fn, (trn_idx, val_idx) in enumerate(KFold(n_splits=NFOLDS, shuffle=True).split(train)):
            print('Fold: ', fn + 1)
            X_train, X_val = train.iloc[trn_idx], train.iloc[val_idx]
            y_train, y_val = y.iloc[trn_idx], y.iloc[val_idx]

            model = XGBRegressor(tree_method='gpu_hist',
                                 min_child_weight=31.58,
                                 learning_rate=0.05,
                                 colsample_bytree=0.65,
                                 gamma=3.69,
                                 max_delta_step=2.07,
                                 max_depth=10,
                                 n_estimators=166,
                                 subsample=0.86)

            model.fit(X_train, y_train)
            pred = model.predict(X_val)
            # pred = [n if n>0 else 0 for n in pred]
            loss = metric(y_val, pred)
            total_loss += loss
            predictions = model.predict(test)
            # predictions = [n if n>0 else 0 for n in predictions]
            submission[column] += predictions / NFOLDS

        submission[column] = submission[column]/NFOLDS
        oof_loss += total_loss / NFOLDS
        print("Model " + str(c) + ": Loss =" + str(total_loss / NFOLDS))
    submission.loc[test['cp_type'] == 1, train_score.columns] = 0
    return submission


def get_xgboost(train, targets, test, NFOLDS=7):
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

    DATA_DIR = '../../../input/'

    train = pd.read_csv(DATA_DIR + 'train_features.csv')
    targets = pd.read_csv(DATA_DIR + 'train_targets_scored.csv')

    test = pd.read_csv(DATA_DIR + 'test_features.csv')
    sub = pd.read_csv(DATA_DIR + 'sample_submission.csv')
    get_xgboost(train, targets, test)