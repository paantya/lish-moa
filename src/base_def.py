# Это заготовка для произвольной модели

def base_model_def(data_dict, hparams, cv, seed=42, optimization=False, verbose=0):
    train = data_dict['train'].copy()
    test = data_dict['test'].copy()
    target = data_dict['target'].copy()
    feature_cols = data_dict['feature_cols']
    target_cols = data_dict['target_cols']

    test_preds = np.zeros((test.shape[0], len(target_cols)))

    for c, column in enumerate(tqdm(target_cols, 'models_one_cols')):
        y = target[column]
        total_loss = 0
        for fn, (trn_idx, val_idx) in enumerate(cv.split(X=train, y=target)):
            # X_train y_train, = train[feature_cols].iloc[trn_idx].values, target[target_cols].iloc[trn_idx].values
            # X_valid, y_valid = train[feature_cols].iloc[val_idx].values, target[target_cols].iloc[val_idx].values
            X_train, X_val = train[feature_cols].iloc[trn_idx], train[feature_cols].iloc[val_idx]
            y_train, y_val = y.iloc[trn_idx], y.iloc[val_idx]

            pass
            ######################
            ## Doo code
            ######################
            # EX:
            # mode.fit(X_train, y_train, )
            # pred = mode.predict(X_val)
            # # pred = [n if n>0 else 0 for n in pred]
            #
            # loss = metric(y_val, pred)
            # total_loss += loss
            # predictions = mode.predict(test)
            # # predictions = [n if n>0 else 0 for n in predictions]
            # submission[column] += predictions / NFOLDS
    if optimization:
        return loss
    else:
        return loss, submission