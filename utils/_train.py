import numpy as np
import torch

from dataset import MyDataset
from sklearn.utils import class_weight


def evaluate_weights(
        train_dataset: MyDataset,
        binary: bool = False
) -> torch.Tensor:
    y = []
    for idx, label in enumerate(train_dataset.get_dataset_status()):
        y = np.append(y, [idx] * label)
    if binary:
        class_weights = (y == 0.).sum() / y.sum()
    else:
        class_weights = class_weight.compute_class_weight('balanced', classes=np.unique(y), y=y)
    return torch.tensor(class_weights, dtype=torch.float32)
