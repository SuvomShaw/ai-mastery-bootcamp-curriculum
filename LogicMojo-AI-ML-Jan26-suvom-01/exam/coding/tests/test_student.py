import math
from pathlib import Path

import pandas as pd
import pytest
import torch
from torch import nn
from torch.utils.data import DataLoader, TensorDataset

from student import (
    TinyCnn,
    build_customer_product_features,
    count_trainable_parameters,
    confusion_matrix,
    conv2d_output_shape,
    macro_f1_from_confusion,
    make_balanced_sampler_weights,
    top_products_by_revenue,
    train_one_epoch,
)


DATA_DIR = Path(__file__).resolve().parents[1] / "data"


def load_events() -> pd.DataFrame:
    events = pd.read_csv(DATA_DIR / "customer_events.csv")
    events["is_returned"] = events["is_returned"].astype(bool)
    return events


def test_build_customer_product_features_aggregates_tabular_events():
    events = load_events()
    result = build_customer_product_features(events)
    result = result.set_index("customer_id")

    expected_columns = {
        "view_count",
        "cart_count",
        "purchase_count",
        "non_returned_purchase_count",
        "gross_revenue",
        "return_rate",
        "unique_products",
        "avg_order_value",
        "cart_to_purchase_rate",
        "view_to_purchase_rate",
    }
    assert expected_columns.issubset(set(result.columns))

    assert result.loc["C001", "view_count"] == 2
    assert result.loc["C001", "cart_count"] == 1
    assert result.loc["C001", "purchase_count"] == 2
    assert result.loc["C001", "non_returned_purchase_count"] == 1
    assert result.loc["C001", "gross_revenue"] == 1200.0
    assert result.loc["C001", "return_rate"] == pytest.approx(0.5)
    assert result.loc["C001", "unique_products"] == 2
    assert result.loc["C001", "avg_order_value"] == 1200.0
    assert result.loc["C001", "cart_to_purchase_rate"] == pytest.approx(1.0)
    assert result.loc["C001", "view_to_purchase_rate"] == pytest.approx(0.5)

    assert result.loc["C004", "view_count"] == 0
    assert result.loc["C004", "cart_count"] == 0
    assert result.loc["C004", "purchase_count"] == 2
    assert result.loc["C004", "gross_revenue"] == 900.0
    assert result.loc["C004", "return_rate"] == pytest.approx(0.5)
    assert result.loc["C004", "cart_to_purchase_rate"] == pytest.approx(0.0)
    assert result.loc["C004", "view_to_purchase_rate"] == pytest.approx(0.0)

    assert result.loc["C005", "purchase_count"] == 0
    assert result.loc["C005", "gross_revenue"] == 0.0
    assert result.loc["C005", "avg_order_value"] == 0.0


def test_top_products_by_revenue_uses_fixed_dataset():
    events = load_events()
    result = top_products_by_revenue(events, top_n=3)

    expected = pd.DataFrame(
        [
            ["P101", 1, 1200.0, 1, 0.0],
            ["P105", 3, 900.0, 1, 0.5],
            ["P103", 1, 800.0, 1, 0.0],
        ],
        columns=[
            "product_id",
            "non_returned_units",
            "gross_revenue",
            "unique_buyers",
            "return_rate",
        ],
    )

    pd.testing.assert_frame_equal(
        result.reset_index(drop=True),
        expected,
        check_dtype=False,
        atol=1e-7,
        rtol=1e-7,
    )


def test_balanced_sampler_weights_inverse_frequency():
    labels = torch.tensor([0, 0, 0, 1, 1, 2])
    weights = make_balanced_sampler_weights(labels)

    assert weights.shape == labels.shape
    assert weights.dtype.is_floating_point
    assert weights[0].item() == pytest.approx(6 / (3 * 3))
    assert weights[3].item() == pytest.approx(6 / (3 * 2))
    assert weights[5].item() == pytest.approx(6 / (3 * 1))


def test_conv2d_output_shape_matches_formula():
    cases = pd.read_csv(DATA_DIR / "conv_cases.csv")
    for row in cases.itertuples(index=False):
        actual = conv2d_output_shape(
            (row.input_h, row.input_w),
            kernel_size=(row.kernel_h, row.kernel_w),
            stride=(row.stride_h, row.stride_w),
            padding=(row.pad_h, row.pad_w),
            dilation=(row.dilation_h, row.dilation_w),
        )
        assert actual == (row.output_h, row.output_w)


def test_tiny_cnn_forward_shape():
    torch.manual_seed(7)
    model = TinyCnn(num_classes=5)
    x = torch.randn(4, 1, 28, 28)
    logits = model(x)

    assert logits.shape == (4, 5)
    assert torch.isfinite(logits).all()
    assert count_trainable_parameters(model) == 5221


def test_train_one_epoch_updates_parameters_and_returns_metrics():
    torch.manual_seed(11)
    model = TinyCnn(num_classes=3)
    x = torch.randn(12, 1, 28, 28)
    y = torch.tensor([0, 1, 2, 1, 0, 2, 2, 1, 0, 0, 2, 1])
    loader = DataLoader(TensorDataset(x, y), batch_size=4, shuffle=False)
    optimizer = torch.optim.SGD(model.parameters(), lr=0.01)
    criterion = nn.CrossEntropyLoss()

    before = [p.detach().clone() for p in model.parameters() if p.requires_grad]
    metrics = train_one_epoch(model, loader, optimizer, criterion, device="cpu")
    after = [p.detach().clone() for p in model.parameters() if p.requires_grad]

    assert set(metrics) == {"loss", "accuracy"}
    assert math.isfinite(metrics["loss"])
    assert 0.0 <= metrics["accuracy"] <= 1.0
    assert any(not torch.equal(a, b) for a, b in zip(before, after))


def test_confusion_matrix_rows_true_columns_predicted():
    y_true = torch.tensor([0, 0, 1, 1, 2, 2, 2])
    y_pred = torch.tensor([0, 1, 1, 2, 2, 0, 2])
    cm = confusion_matrix(y_true, y_pred, num_classes=3)

    expected = torch.tensor(
        [
            [1, 1, 0],
            [0, 1, 1],
            [1, 0, 2],
        ]
    )
    assert torch.equal(cm.cpu(), expected)
    assert macro_f1_from_confusion(cm) == pytest.approx(0.5555556)
