import torch
import torchvision
import cv2
from PIL import Image
from utils import draw_segmentation_map, get_outputs
from torchvision.transforms import transforms as transforms


model = torchvision.models.detection.maskrcnn_resnet50_fpn(pretrained=True, progress=True,
                                                           num_classes=91)
# set the computation device
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
# load the modle on to the computation device and set to eval mode
model.to(device).eval()

transform = transforms.Compose([
    transforms.ToTensor()
])

def procces(image,a,b,name):
    #image = Image.open(image_path).convert('RGB')
# keep a copy of the original image for OpenCV functions and applying masks
    orig_image = image.copy()
# transform the image
    image = transform(image)
# add a batch dimension
    image = image.unsqueeze(0).to(device)
    masks, boxes, labels = get_outputs(image, model, 0.965)
    result = draw_segmentation_map(orig_image, masks, boxes, labels,a,b,name)


    return result