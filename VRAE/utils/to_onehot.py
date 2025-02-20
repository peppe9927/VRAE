import torch

def to_onehot(labels, num_classes, device):
    """
    Converts labels to one-hot encoding.

    Args:
        labels (Tensor): Tensor of labels.
        num_classes (int): Number of classes.
        device (torch.device): Device to place the tensor on.

    Returns:
        Tensor: One-hot encoded labels.
    """
    labels_onehot = torch.zeros(labels.size()[0], num_classes).to(device)
    labels_onehot.scatter_(1, labels.view(-1, 1), 1)
    return labels_onehot