from __future__ import annotations

from typing import Iterable

import pandas as pd
import torch
from torch import nn


def build_customer_product_features(events: pd.DataFrame) -> pd.DataFrame:
    """Build customer-level product interaction features.

    Required input columns:
    customer_id, product_id, event_type, price, quantity, is_returned
    """
    raise NotImplementedError


def top_products_by_revenue(events: pd.DataFrame, top_n: int = 3) -> pd.DataFrame:
    """Return top products by non-returned purchase revenue.

    Output columns:
    product_id, non_returned_units, gross_revenue, unique_buyers, return_rate
    """
    raise NotImplementedError


def make_balanced_sampler_weights(labels: torch.Tensor) -> torch.Tensor:
    """Return one sampling weight per label using inverse class frequency.

    Use this formula for class c:
        total_samples / (num_classes * count_of_class_c)
    """
    raise NotImplementedError


def conv2d_output_shape(
    input_hw: tuple[int, int],
    kernel_size: int | tuple[int, int],
    stride: int | tuple[int, int] = 1,
    padding: int | tuple[int, int] = 0,
    dilation: int | tuple[int, int] = 1,
) -> tuple[int, int]:
    """Return output height and width for a 2D convolution."""
    raise NotImplementedError


class TinyCnn(nn.Module):
    """Small CNN for 28x28 grayscale image classification."""

    def __init__(self, num_classes: int):
        super().__init__()
        raise NotImplementedError

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        raise NotImplementedError


def count_trainable_parameters(model: nn.Module) -> int:
    """Return the number of trainable parameters in a PyTorch model."""
    raise NotImplementedError


def train_one_epoch(
    model: nn.Module,
    dataloader: Iterable,
    optimizer: torch.optim.Optimizer,
    criterion: nn.Module,
    device: torch.device | str,
) -> dict[str, float]:
    """Train for one epoch and return average loss and accuracy."""
    raise NotImplementedError


def confusion_matrix(
    y_true: torch.Tensor,
    y_pred: torch.Tensor,
    num_classes: int,
) -> torch.Tensor:
    """Return a num_classes x num_classes confusion matrix.

    Rows are true labels. Columns are predicted labels.
    """
    raise NotImplementedError


def macro_f1_from_confusion(cm: torch.Tensor) -> float:
    """Compute macro F1 from a confusion matrix.

    Rows are true labels. Columns are predicted labels.
    """
    raise NotImplementedError
