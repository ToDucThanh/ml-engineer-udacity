import json
import os
import torch
import torch.nn as nn
import torchvision.models as models
import torchvision.transforms as transforms

from PIL import Image
import io
import requests
from PIL import ImageFile

ImageFile.LOAD_TRUNCATED_IMAGES = True
JSON_CONTENT_TYPE = 'application/json'
JPEG_CONTENT_TYPE = 'image/jpeg'


def net(device):
    model = models.mobilenet_v2(pretrained=True)

    for param in model.parameters():
        param.requires_grad = False

    model.classifier[1] = nn.Sequential(
        nn.Linear(in_features=1280, out_features=256),
        nn.ReLU(inplace=True),
        nn.Linear(in_features=256, out_features=133)
    )
    model = model.to(device)

    return model


def model_fn(model_dir):
    device = torch.device(
        "cuda" if torch.cuda.is_available() else "cpu"
    )
    model = net(device)

    with open(os.path.join(model_dir, "model.pth"), "rb") as f:
        checkpoint = torch.load(f, map_location=device)
        model.load_state_dict(checkpoint)
    model.eval()

    return model


def input_fn(request_body, content_type=JPEG_CONTENT_TYPE):
    if content_type == JPEG_CONTENT_TYPE:
        return Image.open(io.BytesIO(request_body))

    elif content_type == JSON_CONTENT_TYPE:
        request = json.loads(request_body)
        url = request['url']
        img_content = requests.get(url).content

        return Image.open(io.BytesIO(img_content))


def predict_fn(img_input, model):
    test_transform = transforms.Compose([
        transforms.Resize(size=(224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(
            mean=[0.485, 0.456, 0.406],
            std=[0.229, 0.224, 0.225]
        )
    ])
    img_input = test_transform(img_input)

    with torch.no_grad():
        prediction = model(img_input.unsqueeze(0))

    return prediction