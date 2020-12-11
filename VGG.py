import torch

from torch.nn import Module

from torchvision import models


class VGG(Module):
    def __init__(self):
        super().__init__()

        self.features = models.vgg19(pretrained=True).features

    def forward(self, x):
        select = {
            21: 'conv4_2',  # content layer

            0:  'conv1_1',  # style layer level 1
            5:  'conv2_1',  # style layer level 2
            10: 'conv3_1',  # style layer level 3
            19: 'conv4_1',  # style layer level 4
            28: 'conv5_1'   # style layer level 5
        }

        features = {}

        for i, layer in enumerate(self.features):
            x = layer(x)

            if i in select:
                features[select[i]] = x.squeeze().flatten(start_dim=1)

        return features


def content_loss(features1, features2):
    layer1 = features1['conv4_2']
    layer2 = features2['conv4_2']

    content_loss_ = torch.mean((layer1 - layer2) ** 2) / 2

    return content_loss_


def style_loss(features1, features2, weights):
    layers1 = [features1['conv1_1'],
               features1['conv2_1'],
               features1['conv3_1'],
               features1['conv4_1'],
               features1['conv5_1']]
    layers2 = [features2['conv1_1'],
               features2['conv2_1'],
               features2['conv3_1'],
               features2['conv4_1'],
               features2['conv5_1']]
    weights = [weights['conv1_1'],
               weights['conv2_1'],
               weights['conv3_1'],
               weights['conv4_1'],
               weights['conv5_1']]

    style_loss_ = 0

    for layer1, layer2, weight in zip(layers1, layers2, weights):
        N1, M1 = layer1.size()
        N2, M2 = layer2.size()

        G1, G2 = gram_matrix(layer1) / M1, gram_matrix(layer2) / M2

        style_loss_ += weight * torch.mean((G1 - G2) ** 2) / 4

    return style_loss_


def gram_matrix(layer):
    return torch.mm(layer, layer.t())
