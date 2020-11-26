
import torch

import inspect
import logging
import numpy as np
import torch.nn as nn
import torch.optim as optim
from tqdm.auto import tqdm

from src.loss.loss import SmoothBCEwLogits
from src.models.base import Model
from src.datasets.base import MoADataset, TestDataset
from src.data.process_data import preprocess_data, set_seed
import gc

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


def inference_fn(model, dataloader, device):
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

# run train one model
def run_training(fold, seed, hparams, folds, test, feature_cols, target_cols, num_features, num_targets, target,
                 verbose=False):

    log = logging.getLogger(f"{__name__}.{inspect.currentframe().f_code.co_name}")
    set_seed(seed)

    train = preprocess_data(folds, hparams.model.patch1)
    test_ = preprocess_data(test, hparams.model.patch1)

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
    if hparams.model.train_models:
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

            train_loss = train_fn(model, optimizer, scheduler, loss_tr, trainloader, hparams['device'])
            valid_loss, valid_preds = valid_fn(model, loss_fn, validloader, hparams['device'])
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
                torch.save(model.state_dict(), f"{hparams.path_model}/S{seed}_FOLD{fold}_.pth")

            elif (hparams.model.early_stop == True):

                early_step += 1
                if (early_step >= early_stopping_steps):
                    break

        gc.collect()

        log.debug('')

    # --------------------- PREDICTION---------------------
    x_test = test_[feature_cols].values
    testdataset = TestDataset(x_test)
    testloader = torch.utils.data.DataLoader(testdataset, batch_size=hparams.datamodule.batch_size, shuffle=False)

    model = Model(
        num_features=num_features,
        num_targets=num_targets,
        hidden_size=hparams.model.hidden_size,

    )

    model.load_state_dict(torch.load(f"{hparams['path_model']}/S{seed}_FOLD{fold}_.pth",
                                     map_location=torch.device(hparams['device'])
                                     ))
    model.to(hparams['device'])

    predictions = np.zeros((len(test_), target.iloc[:, 1:].shape[1]))
    predictions = inference_fn(model, testloader, hparams['device'])
    del model
    gc.collect()
    if hparams.model.train_models:
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