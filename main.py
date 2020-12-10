from __future__ import division

import torch

from utils import *
from VGG import VGG


def squared_error_loss(x, y):
    b, c, h, w = x.size()
    F_x = x.view(b * c, h * w)
    F_a = y.view(b * c, h * w)

    F_a = torch.mm(F_a, F_a.t())
    F_x = torch.mm(F_x, F_x.t())
    return torch.mean((F_x - F_a) ** 2) / (4 * c * h * w)


p = load_image("png/content.jpg")
a = load_image("png/style.jpg", p.size)
x = load_image("png/content.jpg")

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

p = PIL_to_tensor(p).to(device)
a = PIL_to_tensor(a).to(device)
x = PIL_to_tensor(x).to(device).requires_grad_(True)

vgg = VGG().to(device).eval()

optimizer = torch.optim.Adam([x], lr=0.25)
scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=200, gamma=0.9)

total_step = 1001

style_weights = {'conv1_1': 0.1,
                 'conv2_1': 0.2,
                 'conv3_1': 0.4,
                 'conv4_1': 0.8,
                 'conv5_1': 1.6}

content_weight = 1
style_weight = 1

for step in range(total_step):
    layers_p = vgg(p)
    layers_a = vgg(a)
    layers_x = vgg(x)

    content_loss = torch.mean((layers_x['conv4_2'] - layers_p['conv4_2']) ** 2)

    style_loss = 0
    for l in style_weights:
        style_layer_loss = style_weights[l] * squared_error_loss(layers_x[l], layers_a[l])

        style_loss += style_layer_loss

    total_loss = content_weight * content_loss + style_weight * style_loss

    optimizer.zero_grad()
    total_loss.backward()
    optimizer.step()
    scheduler.step()

    if step % 100 == 0:
        print('Step {}/{}, Total loss:{:.4f}, Content Loss: {:.4f}, Style Loss: {:.4f}'.
              format(step, total_step, total_loss.item(), content_weight * content_loss, style_weight * style_loss))


x = tensor_to_PIL(x)

show_image(x)

save_image(x, 'target.png')
