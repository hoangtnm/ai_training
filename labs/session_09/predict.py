import os
import sys

import torch
from PIL import Image

from utils import get_device
from utils import get_metadata
from utils import get_net
from utils import get_prediction_class
from utils import preprocess_image

if __name__ == '__main__':
    device = get_device()

    # Training dataset metadata
    _, class_names, class_to_idx = get_metadata(sys.argv[1])
    num_classes = len(class_names)
    idx_to_class = {value: key for key, value in class_to_idx.items()}

    # Data preparation
    image = Image.open(sys.argv[2])

    # Net initialization
    net = get_net(classes=num_classes)
    checkpoint_dict = torch.load(os.path.join('checkpoint', 'checkpoint.pth'),
                                 map_location=device)
    net.load_state_dict(checkpoint_dict['model_state_dict'])
    net.eval()
    net.to(device)

    # Prediction
    image_tensor = preprocess_image(image, mode='val')
    image_tensor = image_tensor.to(device)
    prediction = net(image_tensor)
    result = get_prediction_class(prediction, idx_to_class)
    print(f'Result: {result}')
