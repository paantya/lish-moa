
import torch.nn as nn
import torch.nn.functional as F
from src.layer.recalibrate import recalibrate_layer

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
# model and SmoothingLoss
class DenoisingAutoEncoder(nn.Module):
    def __init__(self, num_features, hidden_size=[1024,2048], dropout=0.2, activation=F.leaky_relu):
        super(DenoisingAutoEncoder, self).__init__()

        self.activation = activation

        self.encoder_list = [nn.Linear(num_features, hidden_size[0]),
                             self.activation(True)]
        for i in range(len(hidden_size) - 1):
            self.encoder_list += [nn.Linear(hidden_size[i], hidden_size[i + 1]),
                                  self.activation(True)]
        self.encoder = nn.Sequential(*self.encoder_list)

        self.decoder_list = []
        for i in range(len(hidden_size) - 1):
            self.decoder_list += [nn.Linear(hidden_size[i+1], hidden_size[i]),
                                  self.activation(True)]
        self.decoder_list += [nn.Linear(hidden_size[0], num_features),
                              self.activation(True)]
        self.decoder = nn.Sequential(*self.decoder_list)

        # self.batch_norm2 = nn.BatchNorm1d(hidden_size)
        # self.dropout2 = nn.Dropout(dropout)
    def forward(self, x):
        encode = self.encoder(x)
        x = self.decoder(encode)
        return encode, x


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
        return x1, x2