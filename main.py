from __future__ import division

import torch

from utils import *
from VGG import VGG


p = load_image("png/content.jpg")
a = load_image("png/style.jpg")
x = load_image("png/content.jpg")

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

p = PIL_to_tensor(p).to(device)
a = PIL_to_tensor(a).to(device)
x = PIL_to_tensor(x).to(device).requires_grad_(True)

vgg = VGG().to(device)

optimizer = torch.optim.Adam([x])

total_step = 1001
style_weight = 100.
for step in range(total_step):
    layers_p = vgg(p)
    layers_a = vgg(a)
    layers_x = vgg(x)

    style_loss = 0
    content_loss = 0
    for F_p, F_a, F_x in zip(layers_p, layers_a, layers_x):
        content_loss += torch.mean((F_x - F_p) ** 2)

        n, m = F_x.size()
        F_a = torch.mm(F_a, F_a.t())
        F_x = torch.mm(F_x, F_x.t())
        style_loss += torch.mean((F_a - F_x) ** 2) / (4 * n * m)

    total_loss = content_loss + style_weight * style_loss

    optimizer.zero_grad()
    total_loss.backward()
    optimizer.step()

    if step % 100 == 0:
        print('Step {}/{}, Content Loss: {:.4f}, Style Loss: {:.4f}'.format(step, total_step, content_loss, style_loss))


x = tensor_to_PIL(x)

show_image(x)

save_image(x, 'target.png')
