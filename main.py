from __future__ import division
from torchvision import models
from torchvision import transforms
from PIL import Image
import torch

import matplotlib.pyplot as plt


class VGG(torch.nn.Module):
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


def load_image(image_fp):
    image = Image.open(image_fp)

    while image.width > 512 or image.height > 512:
        image = image.resize((image.width // 2, image.height // 2))

    transform = transforms.Compose([transforms.ToTensor(),
                                    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])])

    image = transform(image).unsqueeze(0)

    return image


def imshow(tensor):
    plt.ion()
    unloader = transforms.ToPILImage()
    image = tensor.cpu().clone()
    image = image.squeeze(0)
    image = unloader(image)
    plt.imshow(image)
    plt.pause(0.001)


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

content = load_image("png/content.jpg").to(device)
style = load_image("png/style.jpg").to(device)
target = load_image("png/content.jpg").to(device).requires_grad_(True)

vgg = VGG().to(device)

optimizer = torch.optim.Adam([target], lr=0.003, betas=(0.5, 0.999))

total_step = 1001
style_weight = 100.
for step in range(total_step):
    content_features = vgg(content)
    style_features = vgg(style)
    target_features = vgg(target)

    style_loss = 0
    content_loss = 0
    for F_target, F_content, F_style in zip(target_features, content_features, style_features):
        content_loss += torch.mean((F_target - F_content) ** 2)

        n, m = F_content.size()

        F_target = torch.mm(F_target, F_target.t())
        F_style = torch.mm(F_style, F_style.t())
        style_loss += torch.mean((F_target - F_style) ** 2) / (n * m)

    total_loss = content_loss + style_weight * style_loss

    optimizer.zero_grad()
    total_loss.backward()
    optimizer.step()

    if step % 100 == 0:
        print("Step {}/{}, Content Loss: {:.4f}, Style Loss: {:.4f}"
              .format(step, total_step, content_loss.item(), style_loss.item()))

denorm = transforms.Normalize((-2.12, -2.04, -1.80), (4.37, 4.46, 4.44))
img = target.clone().squeeze()
img = denorm(img).clamp_(0, 1)
imshow(img)
