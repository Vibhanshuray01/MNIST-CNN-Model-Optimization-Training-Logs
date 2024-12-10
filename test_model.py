import torch
import pytest
from model import Net
from torchsummary import summary

@pytest.fixture
def model():
    return Net()

def test_parameter_count(model):
    """Test that model has less than 20k parameters"""
    total_params = sum(p.numel() for p in model.parameters())
    assert total_params < 20000, f"Model has {total_params} parameters, should be < 20000"

def test_batch_norm_usage(model):
    """Test that model uses batch normalization"""
    has_batch_norm = any(isinstance(m, torch.nn.BatchNorm2d) for m in model.modules())
    assert has_batch_norm, "Model should use batch normalization"

def test_dropout_usage(model):
    """Test that model uses dropout"""
    has_dropout = any(isinstance(m, torch.nn.Dropout) for m in model.modules())
    assert has_dropout, "Model should use dropout"

def test_gap_usage(model):
    """Test that model uses Global Average Pooling"""
    has_gap = any(isinstance(m, torch.nn.AvgPool2d) for m in model.modules())
    assert has_gap, "Model should use Global Average Pooling"

def test_forward_pass(model):
    """Test that model can process a forward pass"""
    x = torch.randn(1, 1, 28, 28)
    output = model(x)
    assert output.shape == (1, 10), f"Expected output shape (1, 10), got {output.shape}" 