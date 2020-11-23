import os
import pandas as pd
from src.tree.xgb import get_xgboost


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

    print(train.info('all'))
    print(train.head(1))
    get_xgboost(train=train, targets=targets, test=test, NFOLDS=2)

if __name__ == '__main__':
    run()