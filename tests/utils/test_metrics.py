import torch

from UCF_VIT.utils.metrics import DiceBLoss, masked_mse


def test_masked_mse_only_averages_masked_positions():
    pred = torch.tensor([[1.0, 1.0], [3.0, 3.0]])
    y = torch.zeros_like(pred)
    mask = torch.tensor([1.0, 0.0])  # only the first row counts

    loss = masked_mse(pred, y, mask)
    assert loss.item() == 1.0  # mean((1-0)^2) for row 0 == 1.0, row 1 excluded


def test_masked_mse_zero_when_prediction_matches_target():
    pred = torch.randn(4, 5)
    y = pred.clone()
    mask = torch.ones(4)
    loss = masked_mse(pred, y, mask)
    assert loss.item() == 0.0


def test_dice_bloss_near_zero_for_near_perfect_prediction():
    # channel 0 is background and is dropped by DiceBLoss; make channel 1 a
    # confident, correct foreground prediction everywhere.
    targets = torch.zeros(1, 2, 4, 4)
    targets[:, 1] = 1.0

    logits = torch.full((1, 2, 4, 4), -10.0)
    logits[:, 1] = 10.0  # sigmoid(10) ~= 1

    loss = DiceBLoss(weight=0.5)(logits, targets, act=True)
    assert loss.item() < 0.01


def test_dice_bloss_high_for_confidently_wrong_prediction():
    targets = torch.zeros(1, 2, 4, 4)
    targets[:, 1] = 1.0

    logits = torch.full((1, 2, 4, 4), 10.0)
    logits[:, 1] = -10.0  # confidently predicts the opposite of the target

    loss = DiceBLoss(weight=0.5)(logits, targets, act=True)
    assert loss.item() > 1.0


def test_dice_bloss_accepts_probabilities_without_activation():
    targets = torch.zeros(1, 2, 4, 4)
    targets[:, 1] = 1.0
    probs = targets.clone()  # already-perfect probabilities

    loss = DiceBLoss(weight=0.5)(probs, targets, act=False)
    assert loss.item() < 0.01
