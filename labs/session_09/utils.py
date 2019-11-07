import torch
import numpy as np
import matplotlib.pyplot as plt
from torch.utils.data import Dataset
from torch.utils.data import DataLoader
from torchvision.datasets import ImageFolder
from torch.utils.tensorboard import SummaryWriter
from torchvision import transforms


# TODO: Adding a “Projector” to TensorBoard
def select_n_random(data, labels, n=100):
    """Selects n random data points and their corresponding labels from a dataset."""
    assert len(data) == len(labels)

    perm = torch.randperm(len(data))
    return data[perm][:n], labels[perm][:n]


def write_embedding_to_tensorboard(data, targets, feature_size, class_names, writer, global_step=None):
    """Writes embedding to TensorBoard.

    Args:
        data: data points
        targets: corresponding labels
        feature_size:
        class_names (list): list of classes
        writer: TensorBoard writer
        global_step (int): global step value to record
    """
    # select random images and their target indices
    images, labels = select_n_random(data, targets)

    # get the class labels for each image
    class_labels = [class_names[label] for label in labels]

    # log embeddings
    features = images.view(-1, 3 * (feature_size ** 2))
    writer.add_embedding(features,
                         metadata=class_labels,
                         label_img=images,
                         global_step=global_step)


# TODO: Writing loss to TensorBoard
def write_to_tensorboard():
    pass


class EmotionDataset(Dataset):
    """Emotion dataset."""
    # TODO: Custom Emotion Dataset


def get_metadata(path):
    """Returns dataset metadata.

    Args:
        path: PATH to dataset folder.
    
    Returns:
        dataset_sizes: number of total images.
        class_names: A list of classes.
        class_to_idx: A dict mapping class_names to the corresponding labels.

        {'drooling-face': 0,
         'face-savouring-delicious-food': 1,
         'face-with-cowboy-hat': 2,
         ...}
    """
    dataset = ImageFolder(path)
    dataset_size = len(dataset)
    class_names = dataset.classes
    class_to_idx = dataset.class_to_idx
    return dataset_size, class_names, class_to_idx


def get_data_loader(path, batch_size=2, num_workers=2):
    """Returns data_loader.

    Args:
        path: path to dataset folder.
        batch_size: batch_size.
        num_workers: how many sub-processes to use for data loading.
    
    Returns:
        data_loader
    """
    data_transform = get_data_transforms()
    train_set = ImageFolder(root=path, transform=data_transform)
    data_loader = DataLoader(
        train_set, batch_size=batch_size, shuffle=True, num_workers=num_workers)
    return data_loader


def get_data_transforms():
    data_transforms = transforms.Compose([
        transforms.RandomResizedCrop(224),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406],
                             std=[0.229, 0.224, 0.225])
    ])
    return data_transforms


def matplotlib_imshow(img, one_channel=False):
    if one_channel:
        img = img.mean(dim=0)
    img = img / 2 + 0.5  # unnormalize
    img_np = img.cpu().numpy()
    if one_channel:
        plt.imshow(img_np, cmap="Greys")
    else:
        plt.imshow(np.transpose(img_np, (1, 2, 0)))


def images_to_probs(net, images):
    """
    Generates predictions and corresponding probabilities from a trained
    network and a list of images.
    """
    output = net(images).cpu()
    # convert output probabilities to predicted class
    _, preds_tensor = torch.max(output, 1)
    preds = np.squeeze(preds_tensor.numpy())
    return preds, [F.softmax(el, dim=0)[i].item() for i, el in zip(preds, output)]


def plot_classes_preds(net, class_names, images, labels):
    """
    Generates matplotlib Figure using a trained network, along with images
    and labels from a batch, that shows the network's top prediction along
    with its probability, alongside the actual label, coloring this
    information based on whether the prediction was correct or not.
    Uses the "images_to_probs" function.
    """
    preds, probs = images_to_probs(net, images)
    # plot the images in the batch, along with predicted and true labels
    fig = plt.figure(figsize=(12, 48))
    for idx in np.arange(4):
        ax = fig.add_subplot(1, 4, idx + 1, xticks=[], yticks=[])
        matplotlib_imshow(images[idx], one_channel=False)
        ax.set_title("{0}, {1:.1f}%\n(label: {2})".format(
            class_names[preds[idx]],
            probs[idx] * 100.0,
            class_names[labels[idx]]),
            color=("green" if preds[idx] == labels[idx].item() else "red"))
    return fig
