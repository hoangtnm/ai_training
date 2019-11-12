# Google Colab Training
# from google.colab import drive
# drive.mount('/gdrive')
# !mkdir data
# !cp /gdrive/Shared\ drives/AI\ Training/datasets/Dogs\ vs\ Cats/dogs_vs_cats.zip data/
# !unzip data/dogs_vs_cats.zip -d data/cat_vs_dog

import copy
import os
import shutil

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm

from utils import get_device
from utils import get_data_loader
from utils import get_metadata
from utils import get_net
from utils import write_embedding_to_tensorboard


def main(net, checkpoint, feature_size, training_loader, evaluation_loader=None, writer=None, epochs=10, lr=1e-3):
    """Training function.

    Args:
        net: model instance.
        checkpoint: path to checkpoint.
        feature_size:
        training_loader: training data loader.
        evaluation_loader: eval data loader.
        writer: SummaryWriter instance.
        epochs: number of epochs to train the model.
        lr: learning rate.

    Returns:
        net: model instance.
    """
    device = get_device()
    net.to(device)

    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(net.parameters(), lr=lr)

    best_loss = 0.5
    initial_epoch = 0
    if os.path.exists(checkpoint):
        checkpoint = torch.load(checkpoint, map_location=device)
        model.load_state_dict(checkpoint['model_state_dict'])
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        initial_epoch = checkpoint['epoch'] + 1

    for epoch in range(initial_epoch, epochs):
        # print(f'Epoch: {epoch}/{epochs}')
        # time.sleep(1)

        running_loss = 0.0
        running_corrects = 0

        # Iterate over data
        for i, data in enumerate(tqdm(training_loader)):
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

            running_loss += loss.item() * inputs.size(0)
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

        epoch_loss = running_loss / len(training_loader.dataset)
        epoch_acc = running_corrects.double() / len(training_loader.dataset)
        print(f'Epoch: {epoch} - loss: {epoch_loss:.4f} - acc: {epoch_acc:.4f}\n')

        # Saving checkpoint
        if epoch_loss < best_loss:
            best_loss = epoch_loss
            best_net_wts = copy.deepcopy(net.state_dict())
            torch.save({
                'epoch': epoch,
                'loss': epoch_loss,
                'model_state_dict': best_net_wts,
                'optimizer_state_dict': optimizer.state_dict(),
            }, checkpoint)

    return net


if __name__ == '__main__':

    HyperParams = {
        'batch_size': 32,
        'input_size': 224,
        'epochs': 10
    }

    checkpoint_path = os.path.join('checkpoint', 'checkpoint.pth')
    checkpoint_dir = os.path.dirname(checkpoint_path)
    if not os.path.exists(checkpoint_dir):
        os.makedirs(checkpoint_dir)

    # torch.multiprocessing.freeze_support()
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

    num_classes = len(class_names)
    model = get_net(classes=num_classes)
    model = main(model, checkpoint_path, HyperParams['input_size'], train_loader,
                 eval_loader, writer=writer, epochs=HyperParams['epochs'])
    writer.close()
