
import torch.nn as nn
import torch.nn.functional as F

import math
# model and SmoothingLoss
class Model_wn(nn.Module):
    def __init__(self, num_features, num_targets, hidden_size, dropout=0.2):
        super(Model_wn, self).__init__()
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
        x = F.leaky_relu(self.dense1(x))

        x = self.batch_norm2(x)
        x = self.dropout2(x)
        x = F.leaky_relu(self.dense2(x))

        x = self.batch_norm3(x)
        x = self.dropout3(x)
        x = self.dense3(x)

        return x


# model and SmoothingLoss
class Model(nn.Module):
    def __init__(self, num_features, num_targets, hidden_size, dropout=0.2):
        super(Model, self).__init__()
        self.batch_norm1 = nn.BatchNorm1d(num_features)
        self.dense1 = nn.Linear(num_features, hidden_size)

        self.batch_norm2 = nn.BatchNorm1d(hidden_size)
        self.dropout2 = nn.Dropout(dropout)
        self.dense2 = nn.Linear(hidden_size, hidden_size)
        # Model
        self.batch_norm3 = nn.BatchNorm1d(hidden_size)
        self.dropout3 = nn.Dropout(dropout)
        self.dense3 = nn.Linear(hidden_size, num_targets)

    def forward(self, x):
        x = self.batch_norm1(x)
        x = F.leaky_relu(self.dense1(x))

        x = self.batch_norm2(x)
        x = self.dropout2(x)
        x = F.leaky_relu(self.dense2(x))

        x = self.batch_norm3(x)
        x = self.dropout3(x)
        x = self.dense3(x)

        return x


class NetTwoHead(nn.Module):
    def __init__(self, n_in, n_h, n_out, n_out1, loss, rloss):
        super(NetTwoHead, self).__init__()
        self.fc1 = nn.Linear(n_in, n_h)
        self.fc2 = nn.Linear(n_h, math.ceil(n_h / 4))
        self.fc3 = nn.Linear(n_h, math.ceil(n_h / 4))
        self.fc4 = nn.Linear(math.ceil(n_h / 4), n_out)
        self.fc5 = nn.Linear(math.ceil(n_h / 4), n_out1)
        self.bn = nn.BatchNorm1d(n_in)
        self.bn1 = nn.BatchNorm1d(n_h)
        self.bn2 = nn.BatchNorm1d(math.ceil(n_h / 4))
        self.bn3 = nn.BatchNorm1d(math.ceil(n_h / 4))
        self.bn4 = nn.BatchNorm1d(math.ceil(n_h / 4))
        self.drop = nn.Dropout(0.4)
        self.n_out = n_out
        self.selu = nn.SELU()
        self.sigm = nn.Sigmoid()
        self.loss = loss
        self.rloss = rloss

    def forward(self, x, targets, targets1):
        x = self.fc1(self.bn(x))
        x = self.selu(x)
        x = self.fc2(self.drop(self.bn1(x)))
        x = self.selu(x)
        x = self.fc3(self.drop(self.bn2(x)))
        x = self.selu(x)

        # scored targets
        x1 = self.fc3(self.bn3(x))
        # non scored targets
        x2 = self.fc4(self.bn4(x))
        loss = (self.loss(x1, targets) + self.loss(x2, targets1)) / 2
        real_loss = self.rloss(x1, targets)
        # probabilities
        out = self.sigm(x1)
        return out, loss, real_loss