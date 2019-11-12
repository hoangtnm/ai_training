import sys
import os
from functools import reduce

import torch
import torch.nn as nn
from PIL import Image
from torchvision import models

from utils import get_metadata, get_data_transforms, get_device


def predict_all(files, idx_to_class, model_path):
    device = get_device()
    model = models.resnet18()
    model.fc = nn.Linear(512, num_classes)
    checkpoint = torch.load(model_path, map_location=device)
    model.load_state_dict(checkpoint['model_state_dict'])
    model.to(device)

    data_transforms = get_data_transforms()['val']

    tmp = []

    for f in files:
        image = Image.open(f)
        image = data_transforms(image)
        image.unsqueeze_(dim=0)
        image = image.to(device)
        output = model(image)
        target_idx = torch.argmax(output).item()
        target_name = idx_to_class[target_idx]

        head, tail = os.path.split(f)
        tmp.append({
            'fname': tail,
            'prediction': target_name
        })
    return tmp


def analyze(result):
    passed, failed = 0, 0
    for case in result:
        fname = case['fname']
        prediction = case['prediction']
        print(fname, ' --> ', prediction)
        if prediction in fname:
            passed += 1
        else:
            failed += 1
    print('*' * 50)
    print(' > passed: ', passed)
    print(' > failed: ', failed)
    ar = passed / (passed + failed)
    print(' > accuracy ratio: ', '%.2f' % (ar * 100), '%')


def get_files(dname, fpath):
    dpath = os.path.join(fpath, dname)
    return [f'{dpath}/{fname}' for fname in os.listdir(dpath)]


if __name__ == '__main__':
    model_path = sys.argv[1]
    fpath = sys.argv[2]

    # Training dataset metadata
    _, class_names, class_to_idx = get_metadata(fpath)
    num_classes = len(class_names)
    idx_to_class = {value: key for key, value in class_to_idx.items()}

    flist = [get_files(cls, fpath) for cls in class_names]
    files = list(reduce(lambda x, y: x + y, flist))

    result = predict_all(files, idx_to_class, model_path)
    analyze(result)
