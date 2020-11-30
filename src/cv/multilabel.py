import sys
sys.path.append('../../../input/iterative-stratification/iterative-stratification-master')

from iterstrat.ml_stratifiers import MultilabelStratifiedKFold
import pandas as pd
from sklearn.model_selection._split import _BaseKFold

class DrugAwareMultilabelStratifiedKFold(_BaseKFold):

    SAMPLES_PER_EXPERIMENT = 6

    def __init__(self,
                 max_experiment_cnt=3,
                 n_splits=3,
                 shuffle=False,
                 random_state=None):
        super().__init__(n_splits=n_splits,
                         shuffle=shuffle,
                         random_state=random_state)
        self._skf = MultilabelStratifiedKFold(n_splits=n_splits,
                                              shuffle=shuffle,
                                              random_state=random_state)
        self.drug_threshold = self.SAMPLES_PER_EXPERIMENT * max_experiment_cnt

    def _iter_test_indices(self, X=None, y=None, groups=None):
        drug_set = X.merge(y, left_index=True, right_index=True)
        targets = y.columns
        vc = X['drug_id'].value_counts()
        vc1 = vc.loc[vc <= self.drug_threshold].index.sort_values()
        vc2 = vc.loc[vc > self.drug_threshold].index.sort_values()

        drug_id_to_fold = {}
        sig_id_to_fold = {}
        if len(vc1) > 0:
            tmp = drug_set.groupby('drug_id')[targets].mean().loc[vc1]
            for fold, (_, idx_val) in enumerate(self._skf.split(tmp, tmp[targets])):
                drug_id_to_fold.update({k: fold for k in tmp.index[idx_val].values})

        if len(vc2) > 0:
            tmp = drug_set.loc[drug_set.drug_id.isin(vc2)].reset_index()
            for fold, (_, idx_val) in enumerate(self._skf.split(tmp, tmp[targets])):
                sig_id_to_fold.update({k: fold for k in tmp.sig_id[idx_val].values})

        drug_set['fold'] = drug_set.drug_id.map(drug_id_to_fold)
        unset_folds = drug_set.fold.isna()
        drug_set.loc[unset_folds, 'fold'] = drug_set.loc[unset_folds].index.map(sig_id_to_fold)
        test_folds = drug_set.fold.astype('int8').values

        for i in range(self.n_splits):
            yield test_folds == i




# train_targets_scored = pd.read_csv(input_dir + 'train_targets_scored.csv').set_index('sig_id')
# train_drug = pd.read_csv(input_dir + 'train_drug.csv').set_index('sig_id')
# train_features = train_features.merge(train_drug, left_index=True, right_index=True)