import torch
import cv2
import numpy as np
import os
import albumentations as albu
from people_segmentation.pre_trained_models import create_model
from pylab import imshow
from iglovikov_helper_functions.utils.image_utils import load_rgb, pad, unpad
from iglovikov_helper_functions.dl.pytorch.utils import tensor_from_rgb_image


def apply_segmentation(frame, color):
    if color == 'red':
        c = (0, 0, 255)
    elif color == 'green':
        c = (0,255,0)

    model = create_model("Unet_2020-07-20")

    model.eval()

    transform = albu.Compose([albu.Normalize(p=1)], p=1)

    padded_image, pads = pad(frame, factor=32, border=cv2.BORDER_CONSTANT)

    x = transform(image=padded_image)["image"]
    x = torch.unsqueeze(tensor_from_rgb_image(x), 0)

    with torch.no_grad():
        prediction = model(x)[0][0]

    mask = (prediction > 0).cpu().numpy().astype(np.uint8)

    mask = unpad(mask, pads)

    tmp = (cv2.cvtColor(mask, cv2.COLOR_GRAY2RGB) * c).astype(np.uint8)
    tmp[mask == 0] = (255,0,0)
    dst = cv2.addWeighted(frame, 1,tmp, 0.5, 0)

    return dst

