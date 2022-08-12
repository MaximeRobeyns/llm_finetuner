"""Tests for the high-level functionality"""

from finetuna.main import quantize_base_model
import torch as t
import torch.nn as nn

import pytest
import finetuna as ft
import transformers


def test_version_exists():
    assert ft.__version__ is not None


@pytest.mark.slow
def test_load_finetuned():
    model_name = "facebook/opt-125m"
    base_model = transformers.AutoModelForCausalLM.from_pretrained(model_name)

    model_1 = ft.new_finetuned(base_model)
    assert model_1 is not None


def test_param_grads():
    net = nn.Sequential(nn.Linear(5, 20), nn.Linear(20, 1))
    for c in net.children():
        for p in c.parameters():
            assert p.requires_grad == True

    quantize_base_model(net)

    for c in net.children():
        assert isinstance(c, ft.QuantizedLinear)

    for c in net.children():
        for p in c.parameters():
            assert p.requires_grad == False


def test_param_intersection():

    net = nn.Sequential(nn.Linear(5, 20), nn.Linear(20, 1))

    base_set = set(p for p in net.parameters() if p.requires_grad)
    ft_set1 = set(p for p in ft.new_finetuned(net).parameters() if p.requires_grad)
    ft_set2 = set(p for p in ft.new_finetuned(net).parameters() if p.requires_grad)

    assert len(base_set.intersection(ft_set2)) == 0
    assert len(ft_set1.intersection(ft_set2)) == 0
