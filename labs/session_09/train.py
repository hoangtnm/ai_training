# Google Colab Training
# from google.colab import drive
# drive.mount('/gdrive')
# !mkdir data
# !cp /gdrive/My\ Drive/research/datasets/cat_vs_dog/cat_vs_dog.zip data/
# !unzip data/cat_vs_dog.zip -d data/cat_vs_dog


import os
import copy
import time
import shutil

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.tensorboard import SummaryWriter
from torchvision import models

from utils import get_metadata
from utils import get_data_loader
from utils import write_embedding_to_tensorboard
from utils import plot_classes_preds


def main(net, feature_size, training_loader, evaluation_loader=None, writer=None, epochs=10, lr=1e-3):
    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    out_features = len(training_loader.dataset.classes)
    net.fc = nn.Linear(512, out_features)
    net.to(device)

    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(net.parameters(), lr=lr)

    for epoch in range(epochs):
        running_loss = 0.0
        running_corrects = 0

        # Iterate over data
        for i, data in enumerate(training_loader):
            inputs, labels = data
            inputs = inputs.to(device)
            labels = labels.to(device)

            # zero the parameter gradients
            optimizer.zero_grad()

            # forward + backward + optimize
            outputs = net(inputs)
            _, predictions = torch.max(outputs, 1)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            running_loss += loss.item()
            running_corrects += torch.sum(predictions == labels)

            if writer is not None:
                writer.add_scalar('training_loss', loss, epoch * len(training_loader) + i)
            #     writer.add_figure('predictions vs. actual',
            #                       plot_classes_preds(net, class_names, inputs, labels),
            #                       global_step=epoch * len(dataset_loader) + i)

        # Embedding evaluation
        if evaluation_loader is not None:
            images, labels = next(iter(evaluation_loader))
            write_embedding_to_tensorboard(images, labels, feature_size, class_names, writer, epoch)

        epoch_loss = running_loss / len(training_loader)
        epoch_acc = running_corrects.double() / len(training_loader)
        print(f'Epoch: {epoch} - loss: {epoch_loss:.4f} - acc: {epoch_acc:.4f}')

    return net


if __name__ == '__main__':

    HyperParams = {
        'batch_size': 32,
        'input_size': 224,
        'epochs': 10
    }

    torch.multiprocessing.freeze_support()
    dataset_path = os.path.join('data', 'cat_vs_dog', 'train')
    dataset_size, class_names, class_to_idx = get_metadata(dataset_path)

    # TensorBoard setup
    log_path = os.path.join('runs', 'experiment_1')
    if os.path.exists(log_path):
        shutil.rmtree(log_path)
    writer = SummaryWriter(log_path)

    train_loader = get_data_loader(dataset_path, batch_size=HyperParams['batch_size'])
    # eval_loader = get_data_loader(dataset_path, batch_size=dataset_size)
    eval_loader = None

    model = models.resnet18(pretrained=False)
    model = main(model, HyperParams['input_size'], train_loader, eval_loader, writer=writer)
    writer.close()
