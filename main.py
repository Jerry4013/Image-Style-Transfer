import torch

from utils import *
from VGG import VGG, content_loss, style_loss


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

img_c = load_image("input/content.jpg")
img_s = load_image("input/style.jpg")
img_t = load_image("input/content.jpg")

img_c = PIL_to_tensor(img_c).to(device)
img_s = PIL_to_tensor(img_s).to(device)
img_t = PIL_to_tensor(img_t).to(device).requires_grad_(True)

vgg = VGG().to(device).requires_grad_(False).eval()

optimizer = torch.optim.Adam([img_t], lr=0.1)
scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=100, gamma=0.9)

style_layer_weights = {'conv1_1': 0.2,
                       'conv2_1': 0.2,
                       'conv3_1': 0.2,
                       'conv4_1': 0.2,
                       'conv5_1': 0.2}

content_weight = 0.01
style_weight = 99.99

steps = 3000

for step in range(steps):
    features_c = vgg(img_c)
    features_s = vgg(img_s)
    features_t = vgg(img_t)

    content_loss_ = content_loss(features_t, features_c)

    style_loss_ = style_loss(features_t, features_s, style_layer_weights)

    total_loss = content_weight * content_loss_ + style_weight * style_loss_

    if step % 100 == 0:
        print('Step {}/{}, Total loss:{:.4f}, Content Loss: {:.4f}, Style Loss: {:.4f}'.
              format(step, steps, total_loss, content_weight * content_loss_, style_weight * style_loss_))

    optimizer.zero_grad()
    total_loss.backward()
    optimizer.step()
    scheduler.step()

img_t = tensor_to_PIL(img_t)

show_image(img_t)

save_image(img_t, 'output/target.png')
