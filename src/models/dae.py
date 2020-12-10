import torch.nn as nn
import torch.nn.functional as F
from src.layer.recalibrate import recalibrate_layer

import math


# model DenoisingAutoEncoder
class DenoisingAutoEncoder(nn.Module):
    def __init__(self, num_features, hidden_size=[1024, 2048], dropout=0.2, activation=F.leaky_relu):
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
            self.decoder_list += [nn.Linear(hidden_size[i + 1], hidden_size[i]),
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
