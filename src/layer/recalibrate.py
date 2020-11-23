import torch


def recalibrate_layer(self, layer):
    if torch.isnan(layer.weight_v).sum() > 0:
        print('recalibrate layer.weight_v')
        layer.weight_v = torch.nn.Parameter(torch.where(torch.isnan(layer.weight_v),
                                                        torch.zeros_like(layer.weight_v),
                                                        layer.weight_v))
        layer.weight_v = torch.nn.Parameter(layer.weight_v + 1e-7)

    if torch.isnan(layer.weight).sum() > 0:
        print('recalibrate layer.weight')
        layer.weight = torch.where(torch.isnan(layer.weight),
                                   torch.zeros_like(layer.weight),
                                   layer.weight)
        layer.weight += 1e-7
