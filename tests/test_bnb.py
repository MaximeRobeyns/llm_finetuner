from finetuna import quantize_base_model
import torch as t
import torch.nn as nn


def test_quantize():
    net = nn.Sequential(nn.Linear(1, 5), nn.Linear(5, 1))
    x = t.randn(1)
    y = net(x)

    quantize_base_model(net)
    yy = net(x)

    assert t.allclose(y, yy, 1e-2)  # this is conservative
