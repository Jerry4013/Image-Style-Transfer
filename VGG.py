from torch.nn import Module

from torchvision import models


class VGG(Module):
    def __init__(self):
        super().__init__()

        self.vgg = models.vgg19(pretrained=True).features

    def forward(self, x):
        features = []

        for i, layer in enumerate(self.vgg):
            x = layer(x)

            if i in [0, 5, 10, 19, 28]:
                features.append(x.squeeze().flatten(start_dim=1))

        return features
