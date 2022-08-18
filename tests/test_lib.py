"""Tests for the high-level functionality"""

import torch as t
import torch.nn as nn

import pytest
import finetuna as ft
import transformers


class _TestNet(nn.Module):
    def __init__(self):
        super().__init__()
        self.em1 = nn.Embedding(num_embeddings=10, embedding_dim=100)
        self.fc1 = nn.Linear(5, 10)
        self.ln1 = nn.LayerNorm(10)
        self.ac1 = nn.ReLU()
        self.fc2 = nn.Linear(10, 5)
        self.lm_head = nn.Linear(5, 1, False)

    def forward(self, x: t.Tensor) -> t.Tensor:
        return self.lm_head(self.fc2(self.ac1(self.ln1(self.fc1(self.em1(x))))))


def test_version_exists():
    assert ft.__version__ is not None


# @pytest.mark.slow
# def test_load_finetuned():
#     model_name = "facebook/opt-125m"
#     base_model = transformers.AutoModelForCausalLM.from_pretrained(model_name)
#
#     model_1 = ft.new_finetuned(base_model)
#     assert model_1 is not None


def test_copy_mod():
    net = nn.Sequential(nn.Linear(1, 20), nn.Linear(20, 1))
    net.register_buffer("test_buffer", t.randn(10))
    net2 = ft.copy_mod(net)

    for p1, p2 in zip(net.parameters(), net2.parameters()):
        assert p1 is p2

    for b1, b2 in zip(net.buffers(), net2.buffers()):
        assert b1 is b2


def test__is_quantized():
    net = _TestNet()
    assert not ft.main._is_quantized(net)


def test_quantize_base():
    net = _TestNet()
    modules_not_to_quantize = {"fc2", "lm_head"}
    ft.quantize_base_model(net, modules_not_to_quantize)
    comparison = _TestNet()

    # for n, m in comparison.named_modules():
    for mod in comparison.modules():
        for n, m in mod.named_children():

            if isinstance(m, nn.Linear):
                assert isinstance(net._modules[n], ft.FTModule)

                if n.split(".")[-1] in modules_not_to_quantize:
                    # This module is not quantized by default
                    assert type(net._modules[n]) == ft.FTLinear
                else:
                    print(f"doing {m} in {mod} in {net}")
                    assert type(net._modules[n]) == ft.QuantizedLinear

            elif isinstance(m, nn.Embedding):
                assert isinstance(net._modules[n], ft.FTModule)

                if n.split(".")[-1] in modules_not_to_quantize:
                    assert type(net._modules[n]) == ft.FTEmbedding
                else:
                    assert type(net._modules[n]) == ft.QuantizedEmbedding

            elif "LearnedPositionalEmbedding" in str(type(m)):
                assert isinstance(net._modules[n], ft.FTModule)

                if n.split(".")[-1] in modules_not_to_quantize:
                    assert type(net._modules[n]) == ft.FTLearnedPositionalEmbedding
                else:
                    assert (
                        type(net._modules[n]) == ft.QuantizedLearnedPositionalEmbedding
                    )

            else:
                assert not isinstance(net._modules[n], ft.FTModule)


def test_get_adaptable():
    # attempt get_adaptable for non-initialised model should raise exception
    net = _TestNet()

    # only the linear layers
    ref2 = {"fc1", "fc2", "lm_head"}
    mods = ft.get_lora_adaptable_modules(net, [nn.Linear])
    assert ref2 == mods

    # only the (vanilla) Embedding layers
    ref2 = {"em1"}
    mods = ft.get_lora_adaptable_modules(net, [nn.Embedding])
    assert ref2 == mods

    with pytest.raises(RuntimeWarning):
        ft.get_lora_adaptable_modules(net)

    ft.quantize_base_model(net)

    # these are the nn.Linear and nn.Embedding modules in net.
    ref = {"em1", "fc1", "fc2", "lm_head"}
    mods = ft.get_lora_adaptable_modules(net)
    assert ref == mods

    # only the linear layers
    ref2 = {"fc1", "fc2", "lm_head"}
    mods = ft.get_lora_adaptable_modules(net, [ft.FTLinear, ft.QuantizedLinear])
    assert ref2 == mods

    # only the embedding layers
    ref2 = {"em1"}
    mods = ft.get_lora_adaptable_modules(net, [ft.FTEmbedding, ft.QuantizedEmbedding])
    assert ref2 == mods


# Layer initialisations -------------------------------------------------------


def test_init_FTLinear():
    base_linear = nn.Linear(10, 20)
    new_linear = ft.FTLinear.from_linear(base_linear)

    assert new_linear.in_features == base_linear.in_features
    assert new_linear.out_features == base_linear.out_features

    assert new_linear.lora_adapter is None
    assert new_linear.weight is not None

    # Check that they share the same weight tensor
    assert t.all(base_linear.weight.data == new_linear.weight)
    assert base_linear.weight.data.data_ptr() == new_linear.weight.data_ptr()

    # Check that they share the same bias tensor
    assert new_linear.bias is not None
    assert t.all(base_linear.bias.data == new_linear.bias.data)
    assert base_linear.bias.data.data_ptr() == new_linear.bias.data.data_ptr()

    x = t.randn(10)
    assert t.allclose(base_linear(x), new_linear(x))


def test_init_FTLinear_no_bias():
    base_linear = nn.Linear(10, 20, bias=False)
    new_linear = ft.FTLinear.from_linear(base_linear)

    assert new_linear.lora_adapter is None
    assert new_linear.weight is not None
    assert new_linear.bias is None

    # Check that they share the same weight tensor
    assert t.all(base_linear.weight.data == new_linear.weight)
    assert base_linear.weight.data.data_ptr() == new_linear.weight.data_ptr()

    assert new_linear.bias is None

    x = t.randn(10)
    assert t.allclose(base_linear(x), new_linear(x))


def test_init_QuantizedLinear():
    n = 10
    base_linear = nn.Linear(n, 1)
    new_linear = ft.QuantizedLinear.from_linear(base_linear)
    assert new_linear.in_features == base_linear.in_features
    assert new_linear.out_features == base_linear.out_features

    assert not hasattr(new_linear, "weight")
    assert new_linear.quant_weight is not None
    assert new_linear.quant_absmax is not None
    assert new_linear.quant_code is not None

    assert base_linear.bias is not None
    assert new_linear.bias is not None
    assert t.all(base_linear.bias.data == new_linear.bias.data)
    assert base_linear.bias.data.data_ptr() == new_linear.bias.data.data_ptr()

    # TODO: require tighter accuracy than 1e-2?
    x = t.randn(n)
    assert t.allclose(base_linear(x), new_linear(x), 1e-1)


# embedding -------------------------------------------------------------------


def test_init_FTEmbedding():
    num_embeddings = 10
    embedding_dim = 100
    base_embedding = nn.Embedding(num_embeddings, embedding_dim)
    new_embedding = ft.FTEmbedding.from_embedding(base_embedding)

    assert new_embedding.num_embeddings == num_embeddings
    assert new_embedding.embedding_dim == embedding_dim

    assert new_embedding.lora_adapter is None
    assert new_embedding.weight is not None

    # Check that they share the same weight tensor
    assert t.all(base_embedding.weight.data == new_embedding.weight)
    assert base_embedding.weight.data.data_ptr() == new_embedding.weight.data.data_ptr()

    x = t.randint(0, num_embeddings, (10,))
    assert t.allclose(base_embedding(x), new_embedding(x))


def test_init_QuantizedEmbedding():
    num_embeddings = 10
    embedding_dim = 10
    base_embedding = nn.Embedding(num_embeddings, embedding_dim)
    new_embedding = ft.QuantizedEmbedding.from_embedding(base_embedding)

    assert new_embedding.num_embeddings == num_embeddings
    assert new_embedding.embedding_dim == embedding_dim

    assert not hasattr(new_embedding, "weight")
    assert new_embedding.quant_weight is not None
    assert new_embedding.quant_absmax is not None
    assert new_embedding.quant_code is not None

    x = t.randint(0, num_embeddings, (10,))
    # 1e-1 accuracy is frankly pathetic. Don't use 8-bit quantization for
    # embeddings!
    assert t.allclose(base_embedding(x), new_embedding(x), 1e-1)


# LPE -------------------------------------------------------------------------


def test_init_FTLearnedPositionalEmbedding():
    num_embeddings = 10
    embedding_dim = 100
    base_embedding = nn.Embedding(num_embeddings, embedding_dim)
    new_embedding = ft.FTLearnedPositionalEmbedding.from_base(base_embedding)

    assert new_embedding.num_embeddings == num_embeddings - new_embedding.offset
    assert new_embedding.embedding_dim == embedding_dim

    assert new_embedding.lora_adapter is None
    assert new_embedding.weight is not None

    # Check that they share the same weight tensor
    assert t.all(base_embedding.weight.data == new_embedding.weight)
    assert base_embedding.weight.data.data_ptr() == new_embedding.weight.data.data_ptr()


def test_init_QuantizedLearnedPositionalEmbedding():
    num_embeddings = 10
    embedding_dim = 10
    base_embedding = nn.Embedding(num_embeddings, embedding_dim)
    new_embedding = ft.QuantizedLearnedPositionalEmbedding.from_base(base_embedding)

    assert new_embedding.num_embeddings == num_embeddings - new_embedding.offset
    assert new_embedding.embedding_dim == embedding_dim

    assert not hasattr(new_embedding, "weight")
    assert new_embedding.quant_weight is not None
    assert new_embedding.quant_absmax is not None
    assert new_embedding.quant_code is not None


# New Finetuned ---------------------------------------------------------------


def test_new_finetuned_unquantized():
    """Tests calling new_finetuned on an unquantized model"""
    net = _TestNet()  # .train()
    new = ft.new_finetuned(net)

    # net will have been prepared for adapters
    assert type(net.em1) == ft.FTEmbedding
    assert hasattr(net.em1, "lora_adapter")
    assert net.em1.lora_adapter is None

    assert type(net.fc1) == ft.FTLinear
    assert hasattr(net.fc1, "lora_adapter")
    assert net.fc1.lora_adapter is None
    assert net.fc1.bias is not None

    assert type(net.ln1) == nn.LayerNorm
    assert not hasattr(net.ln1, "lora_adapter")

    assert type(net.ac1) == nn.ReLU
    assert not hasattr(net.ac1, "lora_adapter")

    assert type(net.fc2) == ft.FTLinear
    assert hasattr(net.fc2, "lora_adapter")
    assert net.fc2.lora_adapter is None

    assert type(net.lm_head) == ft.FTLinear
    assert hasattr(net.lm_head, "lora_adapter")
    assert net.lm_head.lora_adapter is None
    assert net.lm_head.bias is None

    # -----

    assert type(new.em1) == ft.QuantizedEmbedding
    assert hasattr(new.em1, "lora_adapter")
    # TODO: test the embedding adapter parameters in separate tests
    assert type(new.em1.lora_adapter) == ft.EmbeddingAdapter

    assert type(new.fc1) == ft.QuantizedLinear
    assert hasattr(new.fc1, "lora_adapter")
    assert type(new.fc1.lora_adapter) == ft.LinearAdapter
    assert type(net.fc1.bias) == nn.Parameter
    # shares bias with underlying model:
    assert new.fc1.bias.data.data_ptr() == net.fc1.bias.data.data_ptr()
    assert not hasattr(new.fc1, "weight")  # only has quantized weight
    assert new.fc1.bias.requires_grad == False

    assert type(new.ln1) == nn.LayerNorm
    assert not hasattr(new.ln1, "lora_adapter")

    assert type(new.ac1) == nn.ReLU
    assert not hasattr(new.ac1, "lora_adapter")

    # owing tot he default args of new_finetuned, modules named "lm_head" are
    # treated as plain layers
    assert type(new.lm_head) == ft.FTLinear
    assert hasattr(new.lm_head, "lora_adapter")
    assert new.lm_head.lora_adapter is None
    assert new.lm_head.weight.data.data_ptr() != net.lm_head.weight.data_ptr()


def test_validate_ft_args():
    adaptable_layers = {"em1", "fc1", "fc2", "lm_head"}

    # unquantized model
    net = _TestNet()
    n_adapt_layers, _ = ft.main._validate_new_ft_args(net, None, set(), set(), set())
    assert n_adapt_layers == adaptable_layers

    # quantized model
    ft.quantize_base_model(net, set())
    n_adapt_layers, _ = ft.main._validate_new_ft_args(net, None, set(), set(), set())
    assert n_adapt_layers == adaptable_layers

    # defaults for unquantized net
    net = _TestNet()
    adapt_layers = None
    plain_layers = {"lm_head"}
    modules_not_to_quantize = set()
    modules_not_to_freeze = set()
    n_adapt_layers, quantized_modules = ft.main._validate_new_ft_args(
        net, adapt_layers, plain_layers, modules_not_to_quantize, modules_not_to_freeze
    )
    assert n_adapt_layers == {"em1", "fc1", "fc2"}
    assert quantized_modules == {"em1", "fc1", "fc2"}
    assert modules_not_to_quantize == {"lm_head"}
    assert modules_not_to_freeze == {"lm_head"}

    # defaults for quantized net
    net = _TestNet()
    ft.quantize_base_model(net)
    adapt_layers = None
    plain_layers = {"lm_head"}
    modules_not_to_quantize = set()
    modules_not_to_freeze = set()
    n_adapt_layers, quantized_modules = ft.main._validate_new_ft_args(
        net, adapt_layers, plain_layers, modules_not_to_quantize, modules_not_to_freeze
    )
    assert n_adapt_layers == {"em1", "fc1", "fc2"}
    assert quantized_modules == {"em1", "fc1", "fc2"}
    assert modules_not_to_quantize == {"lm_head"}
    assert modules_not_to_freeze == {"lm_head"}

    # non-defaults for non-quantized net
    net = _TestNet()
    # - will by default convert all possible modules into FTModules and freeze
    # their parameters.
    # - we request to LoRA adapt fc1; which entails quantizing the base params
    adapt_layers = {"fc1"}
    plain_layers = {"lm_head"}
    modules_not_to_quantize = {"em1"}  # don't adapt this, but also don't quantize this
    modules_not_to_freeze = set()
    n_adapt_layers, quantized_modules = ft.main._validate_new_ft_args(
        net, adapt_layers, plain_layers, modules_not_to_quantize, modules_not_to_freeze
    )
    assert n_adapt_layers == {"fc1"}
    assert quantized_modules == {"fc1"}
    assert modules_not_to_quantize == {"em1", "lm_head"}
    assert modules_not_to_freeze == {
        "lm_head"
    }  # em1 and fc2 are still frozen, although not adapted


def test_new_finetuned_default():
    """Tests the new_finetuned function with default arguments.

    In this setting, all modules are quantized, except the lm_head.
    """
    net = _TestNet()
    ft.quantize_base_model(net)
    nf1 = ft.new_finetuned(net)

    assert nf1 is not None

    ll = {"lm_head"}  # non-quantized linear
    ql = {"fc1", "fc2"}  # quantized linear
    el = {}  # non-quantized embedding
    qe = {"em1"}  # quantized embedding
    ml = {"ac1", "ln1"}  # vanilla nn module

    for n, l in nf1.named_children():
        if n in ll:
            assert type(l) == ft.FTLinear
        elif n in ql:
            assert type(l) == ft.QuantizedLinear
        elif n in el:
            assert type(l) == ft.FTEmbedding
        elif n in qe:
            assert type(l) == ft.QuantizedEmbedding
        elif n in ml:
            assert isinstance(l, nn.Module)
            assert not isinstance(l, ft.FTModule)

    # for ((n1, p1), (n2, p2)) in zip(net.named_parameters(), nf1.named_parameters()):
    #     print(n1)
    # assert False


# Gradient tests --------------------------------------------------------------
# TODO: check that various modules have the correct gradients


# def test_FTLinear_grad():
#     base_linear = nn.Linear(10, 20)
#     new_linear = ft.FTLinear.from_linear(base_linear)
#
#     # by default, base model parameters are frozen.
#     # TODO: when creating non-quantized models, we might want to create
#     # non-frozen copies here?
#     assert not any([p.requires_grad for p in new_linear.parameters()])
#
#
# def test_param_grads():
#     net = nn.Sequential(nn.Linear(5, 20), nn.Linear(20, 1))
#     for c in net.children():
#         for p in c.parameters():
#             assert p.requires_grad == True
#
#     ft.quantize_base_model(net)
#
#     # for c in net.children():
#     #     assert isinstance(c, ft.QuantizedLinear)
#
#     # for c in net.children():
#     #     for p in c.parameters():
#     #         assert p.requires_grad == False
#
#
# def test_param_intersection():
#
#     net = nn.Sequential(nn.Linear(5, 20), nn.Linear(20, 1))
#
#     base_set = set(p for p in net.parameters() if p.requires_grad)
#     ft_set1 = set(p for p in ft.new_finetuned(net).parameters() if p.requires_grad)
#     ft_set2 = set(p for p in ft.new_finetuned(net).parameters() if p.requires_grad)
#
#     assert len(base_set.intersection(ft_set2)) == 0
#     assert len(ft_set1.intersection(ft_set2)) == 0
