import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from src.layer.recalibrate import recalibrate_layer



# model and SmoothingLoss
class Model_wn(nn.Module):
    def __init__(self, num_features, num_targets, hidden_size, dropout=0.2):
        super(Model_wn, self).__init__()

        self.num_features = num_features
        self.num_targets = num_targets
        self.hidden_size = hidden_size
        self.dropout = dropout

        self.batch_norm1 = nn.BatchNorm1d(num_features)
        self.dense1 = nn.utils.weight_norm(nn.Linear(num_features, hidden_size))

        self.batch_norm2 = nn.BatchNorm1d(hidden_size)
        self.dropout2 = nn.Dropout(dropout)
        self.dense2 = nn.utils.weight_norm(nn.Linear(hidden_size, hidden_size))
        # Model
        self.batch_norm3 = nn.BatchNorm1d(hidden_size)
        self.dropout3 = nn.Dropout(dropout)
        self.dense3 = nn.utils.weight_norm(nn.Linear(hidden_size, num_targets))

    def forward(self, x):
        x = self.batch_norm1(x)
        x = F.leaky_relu(recalibrate_layer(self.dense1(x)))

        x = self.batch_norm2(x)
        x = self.dropout2(x)
        x = F.leaky_relu(recalibrate_layer(self.dense2(x)))

        x = self.batch_norm3(x)
        x = self.dropout3(x)
        x = recalibrate_layer(self.dense3(x))

        return x


# model and SmoothingLoss
class Model(nn.Module):
    def __init__(self, num_features, num_targets, hidden_size, loss_tr, loss_vl, dropout=0.2):
        super(Model, self).__init__()

        self.num_features = num_features
        self.num_targets = num_targets
        self.hidden_size = hidden_size
        self.loss_tr = loss_tr
        self.loss_vl = loss_vl
        self.dropout = dropout

        self.batch_norm1 = nn.BatchNorm1d(num_features)
        self.dense1 = nn.Linear(num_features, hidden_size)

        self.batch_norm2 = nn.BatchNorm1d(hidden_size)
        self.dropout2 = nn.Dropout(dropout)
        self.dense2 = nn.Linear(hidden_size, hidden_size)
        # Model
        self.batch_norm3 = nn.BatchNorm1d(hidden_size)
        self.dropout3 = nn.Dropout(dropout)
        self.dense3 = nn.Linear(hidden_size, num_targets)

    def forward(self, x, y):
        x = self.batch_norm1(x)
        x = F.leaky_relu(self.dense1(x))

        x = self.batch_norm2(x)
        x = self.dropout2(x)
        x = F.leaky_relu(self.dense2(x))

        x = self.batch_norm3(x)
        x = self.dropout3(x)
        x = self.dense3(x)

        loss = self.loss_tr(x, y)
        loss_real = self.loss_tr(x, y)
        return x, loss, loss_real


class NetTwoHead(nn.Module):
    def __init__(self, n_in, n_h, n_out, n_out1, loss, rloss):
        super(NetTwoHead, self).__init__()
        self.fc1 = nn.Linear(n_in, n_h)
        self.fc11 = nn.Linear(n_h, n_h)
        self.fc2 = nn.Linear(n_h, math.ceil(n_h / 2))
        self.fc3 = nn.Linear(math.ceil(n_h / 2), n_out)
        self.fc4 = nn.Linear(math.ceil(n_h / 2), n_out1)
        self.bn = nn.BatchNorm1d(n_in)
        self.bn1 = nn.BatchNorm1d(n_h)
        self.bn11 = nn.BatchNorm1d(n_h)
        self.bn2 = nn.BatchNorm1d(math.ceil(n_h / 2))
        self.drop = nn.Dropout(0.3)
        self.drop11 = nn.Dropout(0.3)
        self.drop1 = nn.Dropout(0.2)
        self.drop2 = nn.Dropout(0.2)
        self.n_out = n_out
        self.selu = nn.SELU()
        self.sigm = nn.Sigmoid()
        self.loss = loss
        self.rloss = rloss

    def forward(self, x, targets, targets1):
        x = self.fc1(self.bn(x))
        x = F.leaky_relu(x)
        x = self.fc11(self.drop11(self.bn11(x)))
        x = F.leaky_relu(x)
        x = self.fc2(self.drop(self.bn1(x)))
        x = F.leaky_relu(x)

        # scored targets
        x1 = self.fc3(self.drop1(self.bn2(x)))
        # non scored targets
        x2 = self.fc4(self.drop2(self.bn2(x)))
        loss = (self.loss(x1, targets) + self.loss(x2, targets1)) / 2
        rloss = self.rloss(x1, targets)
        out = self.sigm(x1)
        return out, loss, rloss


class TwoHead(nn.Module):
    def __init__(self, num_features, hidden_size, num_targets, num_targets1, loss_tr, loss_vl, two_head_factor=None):
        super(TwoHead, self).__init__()
        if two_head_factor is None:
            self.two_head_factor = [.5, .5]
        else:
            self.two_head_factor = two_head_factor
        self.loss_tr = loss_tr
        self.loss_vl = loss_vl

        self.fc1 = nn.Linear(num_features, hidden_size)
        self.fc11 = nn.Linear(hidden_size, hidden_size)
        self.fc2 = nn.Linear(hidden_size, math.ceil(hidden_size // 2))
        self.fc3 = nn.Linear(math.ceil(hidden_size // 2), num_targets)
        self.fc4 = nn.Linear(math.ceil(hidden_size // 2), num_targets1)
        self.bn = nn.BatchNorm1d(num_features)
        self.bn1 = nn.BatchNorm1d(hidden_size)
        self.bn11 = nn.BatchNorm1d(hidden_size)
        self.bn2 = nn.BatchNorm1d(math.ceil(hidden_size // 2))
        self.drop = nn.Dropout(0.3)
        self.drop11 = nn.Dropout(0.3)
        self.drop1 = nn.Dropout(0.2)
        self.drop2 = nn.Dropout(0.2)
        self.n_out = num_targets
        self.selu = nn.SELU()
        self.sigm = nn.Sigmoid()

    def forward(self, x, y, y1):
        x = self.fc1(self.bn(x))
        x = F.leaky_relu(x)
        x = self.fc11(self.drop11(self.bn11(x)))
        x = F.leaky_relu(x)
        x = self.fc2(self.drop(self.bn1(x)))
        x = F.leaky_relu(x)

        # scored targets
        x1 = self.fc3(self.drop1(self.bn2(x)))
        # non scored targets
        x2 = self.fc4(self.drop2(self.bn2(x)))
        loss = (self.two_head_factor[0] * self.loss_tr(x1, y) +
                self.two_head_factor[1] * self.loss_tr(x2, y1)) / \
               torch.sum(self.two_head_factor)
        loss_real = self.loss_vl(x1, y1)

        return x1, loss, loss_real
