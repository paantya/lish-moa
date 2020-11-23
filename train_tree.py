import os
import pandas as pd
from src.tree.xgb import get_xgboost, get_xgboost1
from src.data.process_data import preprocess_data


def run():

    # print(os.listdir('../input/lish-moa'))
    # print(os.listdir('../'))
    DATA_DIR = '../input/lish-moa/'

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
    get_xgboost1(train=train, targets=targets, test=test, sub=sub, NFOLDS=2)

if __name__ == '__main__':
    run()