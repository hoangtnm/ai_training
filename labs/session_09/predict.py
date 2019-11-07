import sys
import torch
import torch.nn as nn
from PIL import Image
from torchvision import models

from utils import get_metadata
from utils import get_data_transforms


if __name__ == '__main__':
    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

    # Training dataset metadata
    _, class_names, class_to_idx = get_metadata(sys.argv[1])
    num_classes = len(class_names)
    idx_to_class = {value: key for key, value in class_to_idx.items()}

    # Data preparation
    data_transforms = get_data_transforms()
    image = Image.open(sys.argv[2])
    image = data_transforms(image)
    image.unsqueeze_(dim=0)
    image = image.to(device)

    # Prediction
    model = models.resnet18()
    model.fc = nn.Linear(512, num_classes)
    model.to(device)
    output = model(image)
    target_idx = torch.argmax(output).item()
    target_name = idx_to_class[target_idx]
    print(f'Result: {target_name}')
