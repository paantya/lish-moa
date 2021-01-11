
import gc
import os
import torch
import inspect
import logging
import numpy as np
import pandas as pd
import torch.nn as nn
import torch.optim as optim
import pytorch_lightning as pl
from tqdm.auto import tqdm

from math import ceil

from src.loss.loss import SmoothBCEwLogits
from src.models.base import Model, NetTwoHead
from src.datasets.base import MoADataset, TestDataset
from src.datasets.single import MoADatasetSingle
from src.datasets.dual import MoADatasetDual
from src.data.process_data import preprocess_data, set_seed


from hydra.utils import instantiate




# train loop
def train_fn_dual(model, optimizer, scheduler, loss_fn, dataloader, device):
    model.train()
    final_loss = 0

    for data in dataloader:
        optimizer.zero_grad()
        inputs, targets, targets1 = data['data'].to(device), data['target'].to(device), data['target1'].to(device)
        outputs, loss, rloss = model(inputs, targets, targets1)
        # loss = loss_fn(outputs, targets)
        loss.backward()
        optimizer.step()
        scheduler.step()

        final_loss += rloss.item()

    final_loss /= len(dataloader)

    gc.collect()
    return final_loss

# train loop
def train_fn(model, optimizer, scheduler, loss_fn, dataloader, device):
    model.train()
    final_loss = 0

    for data in dataloader:
        optimizer.zero_grad()
        inputs, targets = data['x'].to(device), data['y'].to(device)
        outputs = model(inputs)
        loss = loss_fn(outputs, targets)
        loss.backward()
        optimizer.step()
        scheduler.step()

        final_loss += loss.item()

    final_loss /= len(dataloader)

    gc.collect()
    return final_loss


def valid_fn_dual(model, loss_fn, dataloader, device):
    model.eval()
    final_loss = 0
    valid_preds = []

    for data in dataloader:
        inputs, targets, targets1 = data['data'].to(device), data['target'].to(device), data['target1'].to(device)
        outputs, loss, rloss = model(inputs, targets, targets1)
        # loss = loss_fn(outputs, targets)

        final_loss += rloss.item()
        valid_preds.append(outputs.detach().cpu().numpy())

    final_loss /= len(dataloader)
    valid_preds = np.concatenate(valid_preds)

    gc.collect()
    return final_loss, valid_preds

def valid_fn(model, loss_fn, dataloader, device):
    model.eval()
    final_loss = 0
    valid_preds = []

    for data in dataloader:
        inputs, targets = data['x'].to(device), data['y'].to(device)
        outputs = model(inputs)
        loss = loss_fn(outputs, targets)

        final_loss += loss.item()
        valid_preds.append(outputs.sigmoid().detach().cpu().numpy())

    final_loss /= len(dataloader)
    valid_preds = np.concatenate(valid_preds)

    gc.collect()
    return final_loss, valid_preds

def inference_fn_original(model, dataloader, device):
    model.eval()
    preds = []

    for data in dataloader:
        inputs = data['x'].to(device)

        with torch.no_grad():
            outputs = model(inputs)

        preds.append(outputs.sigmoid().detach().cpu().numpy())

    preds = np.concatenate(preds)

    gc.collect()
    return preds


def inference_fn_dual(model, dataloader, device, batch_size=128):
    model.eval()
    preds = np.zeros((len(dataloader)*batch_size, 206))

    for ind, batch in enumerate(dataloader):
        with torch.no_grad():
            pred = model(batch['data'].to(device), batch['target'].to(device), batch['target1'].to(device))[0].detach().cpu().numpy()
        preds[ind * batch_size: ind * batch_size + pred.shape[0]] = pred
        if pred.shape[0] != batch_size:
            preds = preds[:-(batch_size-pred.shape[0])]

    gc.collect()
    return preds


def inference(model, dataloader, batch_size=128):
    model.to('cpu')
    model.eval()
    preds = np.zeros((len(dataloader)*batch_size, 206))

    for ind, batch in enumerate(dataloader):
        with torch.no_grad():
            pred = model(**{k:v.to('cpu')for k,v in batch.items()})[0].detach().cpu().numpy()
        preds[ind * batch_size: ind * batch_size + pred.shape[0]] = pred
        if pred.shape[0] != batch_size:
            preds = preds[:-(batch_size-pred.shape[0])]
    gc.collect()
    return preds

def inference_fn(model, dataloader, device, batch_size=128):
    model.eval()
    preds = np.zeros((len(dataloader)*batch_size, 206))

    for ind, batch in enumerate(dataloader):
        with torch.no_grad():
            pred = model(batch['x'].to(device)).sigmoid().detach().cpu().numpy()
        preds[ind * batch_size: ind * batch_size + pred.shape[0]] = pred
        if pred.shape[0] != batch_size:
            preds = preds[:-(batch_size-pred.shape[0])]

    gc.collect()
    return preds

# run train one mode
def run_training(fold, seed, hparams, folds, test, feature_cols, target_cols, num_features, num_targets, target,
                 verbose=False):

    log = logging.getLogger(f"{__name__}.{inspect.currentframe().f_code.co_name}")
    set_seed(seed)

    test_ = preprocess_data(test, hparams.model.patch1)
    if hparams.model.train_models:
        train = preprocess_data(folds, hparams.model.patch1)

        trn_idx = train[train['kfold'] != fold].index
        val_idx = train[train['kfold'] == fold].index

        train_df = train[train['kfold'] != fold].reset_index(drop=True)
        valid_df = train[train['kfold'] == fold].reset_index(drop=True)

        x_train, y_train = train_df[feature_cols].values, train_df[target_cols].values
        x_valid, y_valid = valid_df[feature_cols].values, valid_df[target_cols].values
        #     print(f"check sum fold {fold}: train_x={x_train.sum().sum()}, train_y={y_train.sum().sum()}")
        train_dataset = MoADataset(x_train, y_train)
        valid_dataset = MoADataset(x_valid, y_valid)
        trainloader = torch.utils.data.DataLoader(train_dataset, batch_size=hparams.datamodule.batch_size,
                                                  num_workers=hparams.datamodule.num_workers, shuffle=True)
        validloader = torch.utils.data.DataLoader(valid_dataset, batch_size=hparams.datamodule.batch_size,
                                                  num_workers=hparams.datamodule.num_workers, shuffle=False)
        model = Model(
            num_features=num_features,
            num_targets=num_targets,
            hidden_size=hparams.model.hidden_size,
            dropout=hparams.model.dropout_model,
        )

        model.to(hparams['device'])

        optimizer = torch.optim.Adam(model.parameters(), lr=hparams.model.lr,
                                     weight_decay=hparams.model.weight_decay)
        scheduler = optim.lr_scheduler.OneCycleLR(optimizer=optimizer, pct_start=0.1, div_factor=1e3,
                                                  max_lr=1e-2, epochs=hparams.model.epochs,
                                                  steps_per_epoch=len(trainloader))

        loss_fn = nn.BCEWithLogitsLoss()
        loss_tr = SmoothBCEwLogits(smoothing=0.001)

        early_stopping_steps = hparams.model.early_stopping_steps
        early_step = 0

        oof = np.zeros((len(train), target.iloc[:, 1:].shape[1]))
        best_loss = np.inf

        last_valid_loss = 0.0
        for epoch in range(hparams.model.epochs):

            train_loss = train_fn_dual(model, optimizer, scheduler, loss_tr, trainloader, hparams['device'])
            valid_loss, valid_preds = valid_fn_dual(model, loss_fn, validloader, hparams['device'])
            log.debug(f"sd: {seed:>2} fld: {fold:>2}, ep: {epoch:>3}, tr_loss: {train_loss:.6f}, "
                      f"vl_loss: {valid_loss:.6f}, doff_val: {last_valid_loss - valid_loss:>7.1e}")
            last_valid_loss = valid_loss

            if np.isnan(valid_loss):
                log.info(f"valid_loss is nan")
            if valid_loss < best_loss:

                if np.isnan(valid_loss):
                    log.info(f"valid_loss is nan in save models.")

                best_loss = valid_loss
                oof[val_idx] = valid_preds
                torch.save(model.state_dict(), f"{hparams.path_model}/S{seed}FOLD{fold}.pth")

            elif (hparams.model.early_stop == True):

                early_step += 1
                if (early_step >= early_stopping_steps):
                    break

        gc.collect()

        log.debug('')

    # --------------------- PREDICTION---------------------
    x_test = test_[feature_cols].values
    testdataset = TestDataset(x_test)
    testloader = torch.utils.data.DataLoader(testdataset, batch_size=hparams.datamodule.batch_size,
                                             num_workers=hparams.datamodule.num_workers, shuffle=False)

    model = Model(
        num_features=num_features,
        num_targets=num_targets,
        hidden_size=hparams.model.hidden_size,
        dropout=hparams.model.dropout_model,
    )

    model.load_state_dict(torch.load(f"{hparams['path_model']}/S{seed}FOLD{fold}.pth",
                                     map_location=torch.device(hparams['device'])
                                     ))

    model.to(hparams['device'])

    predictions = inference_fn(model, testloader, hparams['device'])
    del model
    gc.collect()
    if hparams.mode.train_models:
        return oof, predictions
    else:

        return predictions


# def run_k_fold_nn_two_head
def run_k_fold_trainer(data_dict, hparams, cv, seed, iseed, prefix='t1', pretrain_model=False, verbose=0):
    log = logging.getLogger(f"{__name__}.{inspect.currentframe().f_code.co_name}")
    set_seed(seed)

    train_features = data_dict['train_features'].copy()
    train_targets_scored = data_dict['train_targets_scored'].copy()
    train_targets_nonscored = data_dict['train_targets_nonscored'].copy()
    train = data_dict['train'].copy()
    test = data_dict['test'].copy()
    target = data_dict['target'].copy()
    feature_cols = data_dict['feature_cols']
    target_cols = data_dict['target_cols']


    oof = np.zeros((len(data_dict['train']), len(data_dict['target_cols'])))
    predictions = np.zeros((len(data_dict['test']), len(data_dict['target_cols'])))

    for fold, (trn_idx, val_idx) in enumerate(tqdm(cv.split(X=train[feature_cols+['drug_id']], y=train[target_cols]),
                                                   f'run {hparams[prefix].nfolds} folds',
                                                   total=hparams[prefix].nfolds,
                                                   leave=False)):

        oof_ = np.zeros((len(train), len(target_cols)))
        # подготовка данных для обучения
        train_valid_data = {
            'train_data': train[feature_cols].iloc[trn_idx].values,
            'train_targets': target[target_cols].iloc[trn_idx].values,
            'valid_data': train[feature_cols].iloc[val_idx].values,
            'valid_targets': target[target_cols].iloc[val_idx].values,
        }

        # подготовка Параметров модели для обучения
        num_features_targets = {
            'num_features': len(feature_cols),
            'num_targets': len(target_cols),
        }

        # ДОбавление под параметров, в случае двухголовой модели
        if hparams[prefix].two_head:
            train_valid_data['train_targets1'] = train_targets_nonscored[train_targets_nonscored.columns].iloc[
                trn_idx].values
            train_valid_data['valid_targets1'] = train_targets_nonscored[train_targets_nonscored.columns].iloc[
                val_idx].values
            num_features_targets['num_targets1'] = train_targets_nonscored.shape[1]
        data_module = instantiate(hparams[prefix].dataloader,
                                  **train_valid_data,
                                  batch_size=hparams[prefix].batch_size,
                                  num_workers=hparams.datamodule.num_workers,
                                  shuffle=True,
                                  )


        if hparams.scheduler._target_.split('.')[-1] == 'OneCycleLR':
            hparams.scheduler['epochs'] = int(hparams[prefix].pl_trainer.min_epochs)
            hparams.scheduler['steps_per_epoch'] = int(ceil(len(trn_idx)/hparams[prefix].batch_size))
            # hparams.scheduler['steps_per_epoch'] = int(2)
        if not pretrain_model:
            # Инициализация модели
            model = instantiate(hparams[prefix].model,
                                **num_features_targets,
                                loss_tr=instantiate(hparams[prefix].loss_tr),
                                loss_vl=instantiate(hparams[prefix].loss_vl),
                                )

            # Инициализация PL_модуля
            pl_module = instantiate(hparams[prefix].pl_modul,
                                    hparams=hparams,
                                    prefix=prefix,
                                    model=model,
                                    )

            # Проверка работоспособности в один проход
            if (iseed == 0 and fold == 0):
                pl.Trainer(
                    **hparams[prefix].pl_trainer,
                    fast_dev_run=True if (iseed==0 and fold==0) else None,
                    weights_summary=None,
                ).tune(model=pl_module, datamodule=data_module)

            # Подбор размера батча
            if hparams['device'] != 'cpu':
                hparams[prefix].pl_trainer['gpus'] = 1
            if hparams[prefix].pl_trainer.gpus > 0 and hparams[prefix].batch_size in ['auto', 'power', 'binsearch']:
                if hparams[prefix].batch_size == 'auto':
                    hparams[prefix].batch_size = 'power'
                pl.Trainer(
                    **hparams[prefix].pl_trainer,
                    auto_scale_batch_size=hparams[prefix].batch_size,
                    weights_summary=None,
                ).tune(model=pl_module, datamodule=data_module)

            # Подбор lr
            if hparams[prefix].lr == 'auto':
                pl.Trainer(
                    **hparams[prefix].pl_trainer,
                    auto_lr_find=True,
                    weights_summary=None,
                ).tune(model=pl_module, datamodule=data_module)

            # Инициализация сохранения topk моделей, пока отключено
            checkpoint_callback = instantiate(hparams.modelcheckpoint,
                                              filepath=f"{hparams['path_model']}/{prefix}S{seed}F{fold}",
                                              # filename=f"{prefix}S{seed}F{fold}",
                                              # dirpath=f"{hparams['path_model']}",
                                              )

            # Инициализация ранней остановки
            callbacks = [instantiate(hparams.earlystopping)]
            # Инициализация гера
            logger = instantiate(hparams.logger,
                                 name=f"{hparams.scheduler._target_.split('.')[-1]}/{hparams[prefix].model._target_.split('.')[-1]}",
                                 version=f"{prefix}S{seed}F{fold}"
                                 )

            # Инициализация рабочего pl тренера
            trainer = pl.Trainer(
                **hparams[prefix].pl_trainer,
                checkpoint_callback=checkpoint_callback,
                callbacks=callbacks,
                logger=logger,
                # filepath=f"{hparams['path_model']}/{prefix}S{seed}F{fold}.pth",
                # weights_save_path=f"{hparams['path_model']}",
                # logger=instantiate(cfg.logger, name=f'test/{}'),
                weights_summary='full' if (iseed==0 and fold==0) else None,
            )
            trainer.fit(model=pl_module, datamodule=data_module)
            torch.save(trainer.get_model().model.state_dict(), f"{hparams['path_model']}/{prefix}S{seed}F{fold}.pth")
            gc.collect()

        # --------------------- PREDICTION---------------------
        if not hparams[prefix].two_head:
            test_dataset = MoADatasetSingle(test[feature_cols].values)
        else:
            test_dataset = MoADatasetDual(test[feature_cols].values)

        test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=hparams[prefix].batch_size,
                                                 num_workers=hparams.datamodule.num_workers, shuffle=False)
        model = instantiate(hparams[prefix].model,
                            **num_features_targets,
                            loss_tr=instantiate(hparams[prefix].loss_tr),
                            loss_vl=instantiate(hparams[prefix].loss_vl),
                            )

        model.load_state_dict(torch.load(f"{hparams['path_model']}/{prefix}S{seed}F{fold}.pth",
                                         map_location=torch.device(hparams['device'])
                                         ))
        model.to('cpu')
        oof_[val_idx] = inference(model,
                                  data_module.val_dataloader(),
                                  batch_size=hparams[prefix].batch_size,
                                  )
        # TODO local check score
        # oof_[val_idx] target[target_cols].iloc[val_idx].values
        pred_ = inference(model, test_loader, batch_size=hparams[prefix].batch_size)
        del model
        gc.collect()
        oof += oof_
        predictions += pred_ / hparams[prefix].nfolds

    gc.collect()
    return oof, predictions


# def run_k_fold_nn_two_head
def run_k_fold_nn_two_head(data_dict, hparams, cv, seed=42, file_prefix='m1', pretrain_model=True, verbose=0):
    log = logging.getLogger(f"{__name__}.{inspect.currentframe().f_code.co_name}")
    set_seed(seed)

    train_features = data_dict['train_features'].copy()
    train_targets_scored = data_dict['train_targets_scored'].copy()
    train_targets_nonscored = data_dict['train_targets_nonscored'].copy()
    train = data_dict['train'].copy()
    test = data_dict['test'].copy()
    target = data_dict['target'].copy()
    feature_cols = data_dict['feature_cols']
    target_cols = data_dict['target_cols']


    oof = np.zeros((len(data_dict['train']), len(data_dict['target_cols'])))
    predictions = np.zeros((len(data_dict['test']), len(data_dict['target_cols'])))

    total_loss = 0
    # for fold, (trn_idx, val_idx) in enumerate(tqdm(cv.split(X=train, y=target),
    #                                                f'run {hparams.mode.nfolds} folds',
    #                                                total=hparams.mode.nfolds,
    #                                                leave=False)):
    # for fold, (_, val) in enumerate(cv.split(X=train_features, y=train_targets_scored)):
    for fold, (trn_idx, val_idx) in enumerate(tqdm(cv.split(X=train[feature_cols+['drug_id']], y=train[target_cols]),
                                                   f'run {hparams.mode.nfolds} folds',
                                                   total=hparams.mode.nfolds,
                                                   leave=False)):
        if not pretrain_model:
            X_train, y_train = train[feature_cols].iloc[trn_idx].values, target[target_cols].iloc[trn_idx].values
            X_valid, y_valid = train[feature_cols].iloc[val_idx].values, target[target_cols].iloc[val_idx].values
            y1_train = train_targets_nonscored[train_targets_nonscored.columns].iloc[trn_idx].values
            y1_valid = train_targets_nonscored[train_targets_nonscored.columns].iloc[val_idx].values

            train_dataset = MoADatasetDual(X_train, y_train, y1_train)
            valid_dataset = MoADatasetDual(X_valid, y_valid, y1_valid)
            trainloader = torch.utils.data.DataLoader(train_dataset, batch_size=hparams.datamodule.batch_size,
                                                      num_workers=hparams.datamodule.num_workers, shuffle=True)
            validloader = torch.utils.data.DataLoader(valid_dataset, batch_size=hparams.datamodule.batch_size,
                                                      num_workers=hparams.datamodule.num_workers, shuffle=False)
            model = NetTwoHead(
                n_in=len(feature_cols),
                n_h=hparams.model.hidden_size,
                n_out=len(target_cols),
                n_out1=train_targets_nonscored.shape[1],
                loss=nn.BCEWithLogitsLoss(),
                rloss=SmoothBCEwLogits(smoothing=0.001)
            )

            model.to(hparams['device'])

            optimizer = torch.optim.Adam(model.parameters(), lr=hparams.model.lr*2,
                                         weight_decay=hparams.model.weight_decay)
            scheduler = optim.lr_scheduler.OneCycleLR(optimizer=optimizer, pct_start=0.1, div_factor=1e3,
                                                      max_lr=1.5e-2, epochs=int(hparams.model.epochs),
                                                      steps_per_epoch=len(trainloader))

            loss_fn = nn.BCEWithLogitsLoss()
            loss_tr = SmoothBCEwLogits(smoothing=0.001)

            early_stopping_steps = hparams.model.early_stopping_steps
            early_step = 0

            oof_ = np.zeros((len(train), len(target_cols)))
            best_loss = np.inf

            last_valid_loss = 0.0
            for epoch in range(int(hparams.model.epochs)):

                train_loss = train_fn_dual(model, optimizer, scheduler, loss_tr, trainloader, hparams['device'])
                valid_loss, valid_preds = valid_fn_dual(model, loss_fn, validloader, hparams['device'])
                log.debug(f"sd: {seed:>2} fld: {fold:>2}, ep: {epoch:>3}, tr_loss: {train_loss:.6f}, "
                          f"vl_loss: {valid_loss:.6f}, doff_val: {last_valid_loss - valid_loss:>7.1e}")
                if verbose:
                    print(f"sd: {seed:>2} fld: {fold:>2}, ep: {epoch:>3}, tr_loss: {train_loss:.6f}, "
                          f"vl_loss: {valid_loss:.6f}, doff_val: {last_valid_loss - valid_loss:>7.1e}")
                last_valid_loss = valid_loss

                if np.isnan(valid_loss):
                    log.info(f"valid_loss is nan")
                if valid_loss < best_loss:

                    if np.isnan(valid_loss):
                        log.info(f"valid_loss is nan in save models.")

                    best_loss = valid_loss
                    oof_[val_idx] = valid_preds
                    torch.save(model.state_dict(), f"{hparams.path_model}/{file_prefix}S{seed}FOLD{fold}.pth")

                elif (hparams.model.early_stop == True):

                    early_step += 1
                    if (early_step >= early_stopping_steps):
                        break

            gc.collect()

            if verbose:
                print('')
            log.debug('')

        # --------------------- PREDICTION---------------------
        testdataset = MoADatasetDual(test[feature_cols].values)
        testloader = torch.utils.data.DataLoader(testdataset, batch_size=hparams.datamodule.batch_size,
                                                 num_workers=hparams.datamodule.num_workers, shuffle=False)

        model = NetTwoHead(
            n_in=len(feature_cols),
            n_h=hparams.model.hidden_size,
            n_out=len(target_cols),
            n_out1=train_targets_nonscored.shape[1],
            loss=nn.BCEWithLogitsLoss(),
            rloss=SmoothBCEwLogits(smoothing=0.001)
        )

        model.load_state_dict(torch.load(f"{hparams['path_model']}/{file_prefix}S{seed}FOLD{fold}.pth",
                                         map_location=torch.device(hparams['device'])
                                         ))

        model.to(hparams['device'])

        pred_ = inference_fn_dual(model, testloader, hparams['device'])
        del model
        gc.collect()

        if not pretrain_model:
            oof += oof_
        predictions += pred_ / hparams.model.nfolds

    gc.collect()
    if not pretrain_model:
        return oof, predictions
    else:
        return predictions


# def run_k_fold_nn
def run_k_fold_nn(data_dict, hparams, cv, seed=42, file_prefix='m1', pretrain_model=True, verbose=0):
    log = logging.getLogger(f"{__name__}.{inspect.currentframe().f_code.co_name}")
    set_seed(seed)

    train_features = data_dict['train_features'].copy()
    train_targets_scored = data_dict['train_targets_scored'].copy()
    train = data_dict['train'].copy()
    test = data_dict['test'].copy()
    target = data_dict['target'].copy()
    feature_cols = data_dict['feature_cols']
    target_cols = data_dict['target_cols']


    oof = np.zeros((len(data_dict['train']), len(data_dict['target_cols'])))
    predictions = np.zeros((len(data_dict['test']), len(data_dict['target_cols'])))

    total_loss = 0
    # for fold, (trn_idx, val_idx) in enumerate(tqdm(cv.split(X=train, y=target),
    #                                                f'run {hparams.mode.nfolds} folds',
    #                                                total=hparams.mode.nfolds,
    #                                                leave=False)):
    # for fold, (_, val) in enumerate(cv.split(X=train_features, y=train_targets_scored)):
    for fold, (trn_idx, val_idx) in enumerate(tqdm(cv.split(X=train[feature_cols+['drug_id']], y=train[target_cols]),
                                                   f'run {hparams.model.nfolds} folds',
                                                   total=hparams.model.nfolds,
                                                   leave=False)):
        if not pretrain_model:
            X_train, y_train, = train[feature_cols].iloc[trn_idx].values, target[target_cols].iloc[trn_idx].values
            X_valid, y_valid = train[feature_cols].iloc[val_idx].values, target[target_cols].iloc[val_idx].values

            train_dataset = MoADataset(X_train, y_train)
            valid_dataset = MoADataset(X_valid, y_valid)
            trainloader = torch.utils.data.DataLoader(train_dataset, batch_size=hparams.datamodule.batch_size,
                                                      num_workers=hparams.datamodule.num_workers, shuffle=True)
            validloader = torch.utils.data.DataLoader(valid_dataset, batch_size=hparams.datamodule.batch_size,
                                                      num_workers=hparams.datamodule.num_workers, shuffle=False)
            model = Model(
                num_features=len(feature_cols),
                num_targets=len(target_cols),
                hidden_size=hparams.model.hidden_size,
                dropout=hparams.model.dropout_model,
            )

            model.to(hparams['device'])

            optimizer = torch.optim.Adam(model.parameters(), lr=hparams.model.lr,
                                         weight_decay=hparams.model.weight_decay)
            print(f"steps_per_epoch=len(trainloader): {len(trainloader)}")
            scheduler = optim.lr_scheduler.OneCycleLR(optimizer=optimizer, pct_start=0.1, div_factor=1e3,
                                                      max_lr=1e-2, epochs=hparams.model.epochs,
                                                      steps_per_epoch=len(trainloader))

            loss_fn = nn.BCEWithLogitsLoss()
            loss_tr = SmoothBCEwLogits(smoothing=0.001)

            early_stopping_steps = hparams.model.early_stopping_steps
            early_step = 0

            oof_ = np.zeros((len(train), len(target_cols)))
            best_loss = np.inf

            last_valid_loss = 0.0
            for epoch in range(hparams.model.epochs):

                train_loss = train_fn(model, optimizer, scheduler, loss_tr, trainloader, hparams['device'])
                valid_loss, valid_preds = valid_fn(model, loss_fn, validloader, hparams['device'])
                log.debug(f"sd: {seed:>2} fld: {fold:>2}, ep: {epoch:>3}, tr_loss: {train_loss:.6f}, "
                          f"vl_loss: {valid_loss:.6f}, doff_val: {last_valid_loss - valid_loss:>7.1e}")
                if verbose:
                    print(f"sd: {seed:>2} fld: {fold:>2}, ep: {epoch:>3}, tr_loss: {train_loss:.6f}, "
                          f"vl_loss: {valid_loss:.6f}, doff_val: {last_valid_loss - valid_loss:>7.1e}")
                last_valid_loss = valid_loss

                if np.isnan(valid_loss):
                    log.info(f"valid_loss is nan")
                if valid_loss < best_loss:

                    if np.isnan(valid_loss):
                        log.info(f"valid_loss is nan in save models.")

                    best_loss = valid_loss
                    oof_[val_idx] = valid_preds
                    torch.save(model.state_dict(), f"{hparams.path_model}/{file_prefix}S{seed}FOLD{fold}.pth")

                elif (hparams.model.early_stop == True):

                    early_step += 1
                    if (early_step >= early_stopping_steps):
                        break

            gc.collect()

            if verbose:
                print('')
            log.debug('')

        # --------------------- PREDICTION---------------------
        testdataset = TestDataset(test[feature_cols].values)
        testloader = torch.utils.data.DataLoader(testdataset, batch_size=hparams.datamodule.batch_size,
                                                 num_workers=hparams.datamodule.num_workers, shuffle=False)

        model = Model(
            num_features=len(feature_cols),
            num_targets=len(target_cols),
            hidden_size=hparams.model.hidden_size,
            dropout=hparams.model.dropout_model,
        )

        model.load_state_dict(torch.load(f"{hparams['path_model']}/{file_prefix}S{seed}FOLD{fold}.pth",
                                         map_location=torch.device(hparams['device'])
                                         ))

        model.to(hparams['device'])

        pred_ = inference_fn(model, testloader, hparams['device'])
        del model
        gc.collect()

        if not pretrain_model:
            oof += oof_
        predictions += pred_ / hparams.model.nfolds

    gc.collect()
    if not pretrain_model:
        return oof, predictions
    else:
        return predictions





# def run_k_fold
def run_k_fold(NFOLDS, seed, hparams, folds, train, test, feature_cols, target_cols, num_features,
                                       num_targets, target, verbose):

    log = logging.getLogger(f"{__name__}.{inspect.currentframe().f_code.co_name}")
    oof = np.zeros((len(train), len(target_cols)))
    predictions = np.zeros((len(test), len(target_cols)))

    for fold in tqdm(range(NFOLDS), 'run_k_fold', leave=verbose):
        return_run = run_training(fold, seed, hparams, folds, test, feature_cols, target_cols, num_features,
                                       num_targets, target, verbose)
        if hparams.model.train_models:
            oof_, pred_ = return_run
            oof += oof_
        else:
            pred_ = return_run
        predictions += pred_ / NFOLDS

    gc.collect()
    if hparams.model.train_models:
        return oof, predictions
    else:
        return predictions