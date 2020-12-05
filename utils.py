import matplotlib.pyplot as plt

from PIL import Image
from torchvision import transforms


def load_image(image_fp):
    image = Image.open(image_fp)

    while image.width > 512 or image.height > 512:
        image = image.resize((image.width // 2, image.height // 2))

    return image


def show_image(image):
    plt.axis('off')
    plt.imshow(image)
    plt.show()


def save_image(image, image_fp):
    image.save(image_fp)


def PIL_to_tensor(image):
    image_ = image.copy()

    image_ = transforms.ToTensor()(image_)

    image_ = image_.unsqueeze(0)

    image_ = normalize(image_)

    return image_


def tensor_to_PIL(image):
    image_ = image.clone().detach()

    image_ = denormalize(image_)

    image_ = image_.squeeze()

    image_ = transforms.ToPILImage()(image_)

    return image_


def normalize(image):
    image_ = transforms.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225))(image)

    return image_


def denormalize(image):
    image_ = transforms.Normalize(mean=(-2.118, -2.036, -1.804), std=(4.367, 4.464, 4.444))(image)

    image_ = image_.clamp(0, 1)

    return image_
