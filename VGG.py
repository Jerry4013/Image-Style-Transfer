from torch.nn import Module

from torchvision import models


class VGG(Module):
    def __init__(self):
        super().__init__()

        self.vgg = models.vgg19(pretrained=True).features

        for parameter in self.vgg.parameters():
            parameter.requires_grad_(False)

    def forward(self, x):
        select = {'0':  'conv1_1',
                  '5':  'conv2_1',
                  '10': 'conv3_1',
                  '19': 'conv4_1',
                  '28': 'conv5_1',
                  '21': 'conv4_2'}

        features = {}

        for name, layer in self.vgg._modules.items():
            x = layer(x)

            if name in select:
                features[select[name]] = x

        return features
