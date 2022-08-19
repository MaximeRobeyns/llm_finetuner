from __future__ import annotations

import io
import copy
import math
import torch as t
import torch.nn as nn
import torch.nn.functional as F
import warnings
import contextlib

from abc import abstractmethod
from enum import Enum
from typing import (
    Any,
    Type,
    Tuple,
    Optional,
    Union,
    Dict,
    Set,
    List,
)  # for Python 3.8 compat
from dataclasses import dataclass

# from torch.cuda.amp import custom_fwd, custom_bwd


with warnings.catch_warnings():
    warnings.simplefilter("ignore")
    with contextlib.redirect_stdout(io.StringIO()):
        from bitsandbytes.functional import quantize_blockwise, dequantize_blockwise
        from bitsandbytes.nn import Linear8bitLt, StableEmbedding

# from bitsandbytes.nn.modules import Linear8bitLt, StableEmbedding, Int8Params


# Utility functions -----------------------------------------------------------


def _is_quantized(model: nn.Module) -> bool:
    try:
        q = model._is_quantized.item()
        assert isinstance(q, bool)
        return q
    except AttributeError:
        return False


def _get_ft_mod_device(mod: FTModule) -> t.device:
    if isinstance(mod, QuantizedModule):
        return next(mod.buffers()).device
    else:
        return next(mod.parameters()).device


def get_model_mem_consumption(model) -> int:
    """Returns an estimate of the model's memory consumption by looking at its
    parameters."""
    mem_params = sum(
        [param.nelement() * param.element_size() for param in model.parameters()]
    )
    mem_bufs = sum([buf.nelement() * buf.element_size() for buf in model.buffers()])
    return mem_params + mem_bufs


def quantize_blockwise_lowmem(
    matrix: nn.Parameter, chunk_size: int = 2**20
) -> Tuple[t.ByteTensor, t.FloatTensor, t.IntTensor]:
    """Breaks the `matrix` quantization into `chunk_size` chunks to lower the
    amount of data that must be simultaneously loaded into memory.

    Args:
        matrix: the matrix to quantize
        chunk_size: the block size to quantize at one time
    """
    # the lowmem chunks must be divisible by the size of the quantization blocks
    assert chunk_size % 4096 == 0
    code = None
    chunks = []
    absmaxes = []

    orig_dtype = matrix.data.dtype

    try:
        flat_tensor = matrix.data.float().view(-1)
    except Exception:
        flat_tensor = matrix.data.float().contiguous().view(-1)

    for i in range((matrix.numel() - 1) // chunk_size + 1):
        input_chunk = flat_tensor[i * chunk_size : (i + 1) * chunk_size].clone()
        quantized_chunk, (absmax_chunk, code) = quantize_blockwise(
            input_chunk, code=code
        )
        chunks.append(quantized_chunk)
        absmaxes.append(absmax_chunk)

    matrix_i8 = t.cat(chunks).reshape_as(matrix)
    absmax = t.cat(absmaxes)
    matrix.data.to(dtype=orig_dtype)
    return matrix_i8, (absmax, code)


def copy_mod(mod: nn.Module) -> nn.Module:
    tmp = copy.copy(mod)
    tmp._parameters = copy.copy(mod._parameters)
    tmp._buffers = copy.copy(mod._buffers)
    tmp._modules = {name: copy_mod(child) for (name, child) in mod._modules.items()}
    return tmp


class ConfigType(Enum):
    linear = 1
    embedding = 2


def _get_config(config: Union[Dict[str, Any], Any], key: str, config_type) -> Any:

    if isinstance(config, dict):
        config = config.get(key)
        if config is None:
            if config_type is ConfigType.linear:
                raise ValueError(f"No LinearAdapterConfig provided for layer {key}")
            elif config_type is ConfigType.embedding:
                raise ValueError(f"No EmbeddingAdapterConfig provided for layer {key}")
    return config


def plain_copy(mod: FTModule) -> FTModule:
    new_mod = copy.deepcopy(mod)
    new_mod.lora_adapter = None  # should be unnecessary
    new_mod.requires_grad_(True)
    return new_mod


# Component classes ------------------------------------------------------------


@dataclass
class EmbeddingAdapterConfig:
    r: int = 4
    alpha: int = 1


@dataclass
class LinearAdapterConfig:
    r: int = 4
    alpha: int = 1
    dropout: float = 0.0
    bias: bool = True


class FTModule(nn.Module):
    def __init__(self):
        """A module which optionally accomodates LoRA adapters."""
        super(FTModule, self).__init__()
        self.lora_adapter: Optional[nn.Module] = None

    @abstractmethod
    def _forward(self, x: t.Tensor, **kwargs) -> t.Tensor:
        """This method must implement the forward pass for the module being
        replaced."""
        pass

    def forward(self, *args, **kwargs) -> t.Tensor:
        """Adds adapter output to the normal layer output, if a LoRA adapter
        has been added for this layer."""
        output = self._forward(*args, **kwargs)
        if self.lora_adapter:
            return self.lora_adapter(args[0]) + output
        return output


class QuantizedModule(FTModule):
    def __init__(
        self,
        **kwargs,
    ):
        """A frozen, 8-bit quantized module which accomodates LoRA adapters."""
        # Through MRO, this resolves to either FTLinear or FTEmbedding constructors.
        super(QuantizedModule, self).__init__(**kwargs)


# Layers ======================================================================
# Classes for layers that we can quantize and/or adapt.


# Linear ----------------------------------------------------------------------


class FTLinear(FTModule):
    def __init__(
        self,
        weight: nn.Parameter,
        bias: Optional[nn.Parameter] = None,
        save_weight: bool = True,
    ):
        super(FTLinear, self).__init__()
        self.out_features, self.in_features = weight.shape

        if bias is not None:
            assert isinstance(bias, nn.Parameter)
        self.register_parameter("bias", bias)

        if save_weight:  # avoids unnecessary saving when used in quantized layer
            assert isinstance(weight, nn.Parameter)
            self.register_parameter("weight", weight)

    def _forward(self, x: t.Tensor, **_) -> t.Tensor:
        return F.linear(x, self.weight, self.bias)

    # def requires_grad_(self, requires_grad: bool = True) -> None:
    #     if "weight" in self._parameters.keys():
    #         self._parameters["weight"].requires_grad_(requires_grad)
    #     # if "bias" in self._parameters.keys():
    #     if self.bias is not None:
    #         self._parameters["bias"].requires_grad_(requires_grad)

    @classmethod
    def from_linear(cls, linear: nn.Linear) -> FTLinear:
        """Converts a PyTorch nn.Linear layer into a linear layer which can
        accomodate a LoRA adapter."""
        return cls(linear.weight, linear.bias)

    def quantized_copy(self) -> QuantizedLinear:
        # tmp = copy_mod(self)
        # assert tmp.lora_adapter is None
        # if self.bias is not None:
        #     tmp._parameters["bias"] = nn.Parameter(self.bias.clone())
        return QuantizedLinear.from_linear(self)


class QuantizedLinear(QuantizedModule, FTLinear):
    def __init__(
        self,
        weight: nn.Parameter,
        bias: Optional[nn.Parameter] = None,
        has_fp16_weights: bool = True,
    ):
        """
        Args:
            has_fp16_weights: set to True if you want to train this quantized
                layer, or False otherwise (i.e. if it will be frozen / have a
                LoRA adapter).
        """
        super(QuantizedLinear, self).__init__(
            weight=weight, bias=bias, save_weight=False
        )
        self._qlinear = Linear8bitLt(
            self.in_features,
            self.out_features,
            bias=bias is not None,
            threshold=6.0,
            has_fp16_weights=has_fp16_weights,
        )
        self._qlinear.load_state_dict(
            {"weight": weight.requires_grad_(False)}, strict=False
        )
        if bias is not None:
            self._qlinear.load_state_dict(
                {"bias": bias.requires_grad_(False)}, strict=False
            )
        self._qlinear = self._qlinear.cuda().half()

    def _forward(self, x: t.Tensor, **_) -> t.Tensor:
        output = self._qlinear(x.half())
        output = output.float()
        return output

    @classmethod
    def from_linear(
        cls, linear: Union[nn.Linear, FTLinear], has_fp16_weights: bool = True
    ) -> QuantizedLinear:
        """Converts a PyTorch nn.Linear (or FTLinear) module to a quantized
        8-bit linear layer with optionally frozen body parameters, which can
        accomodate a LoRA adapter"""
        return cls(linear.weight, linear.bias, has_fp16_weights)


# Embedding Layers ------------------------------------------------------------


class FTEmbedding(FTModule):
    def __init__(self, weight: nn.Parameter, save_weight: bool = True):
        super(FTEmbedding, self).__init__()
        self.num_embeddings, self.embedding_dim = weight.shape

        if save_weight:
            self.register_parameter("weight", weight)

    def _forward(self, x: t.Tensor, **kwargs) -> t.Tensor:
        # NOTE: for models with gradient_checkpointing, since x is an IntTensor
        # and weight may be frozen (requires_grad = False), the resulting
        # output will have requires_grad = False which causes problems with
        # checkpointing.
        return F.embedding(x, self.weight, **kwargs).requires_grad_(True)

    @classmethod
    def from_embedding(cls, embedding: nn.Embedding) -> FTEmbedding:
        """Converts a PyTorch nn.Embedding module to one which accomodates an
        adapter."""
        return cls(embedding.weight)

    def quantized_copy(self) -> QuantizedEmbedding:
        # no need to deepcopy the current layer since weight will be copied
        # when quantized.
        return QuantizedEmbedding.from_embedding(self)


class QuantizedEmbedding(QuantizedModule, FTEmbedding):
    def __init__(self, weight: nn.Parameter):
        super(QuantizedEmbedding, self).__init__(weight=weight, save_weight=False)
        self._qembedding = StableEmbedding(self.num_embeddings, self.embedding_dim)
        self._qembedding.load_state_dict({"weight": weight})

    def _forward(self, x: t.Tensor, **kwargs) -> t.Tensor:
        # See FTEmbedding for note on requires_grad.
        qembed = self._qembedding(x, **kwargs).requires_grad_(True).float()
        assert not t.any(qembed.isnan()), "nans in embedding"
        return qembed

    @classmethod
    def from_embedding(
        cls, embedding: Union[nn.Embedding, FTEmbedding]
    ) -> QuantizedEmbedding:
        """Converts a PyTorch nn.Embedding module to a quantized 8-bit version
        with optionally frozen weights and an optional adapter."""
        return cls(embedding.weight)


# LearnedPositionalEmbedding --------------------------------------------------
# These are specific to facebook/OPT-class models.


class FTLearnedPositionalEmbedding(FTModule):
    def __init__(self, weight: t.Tensor, save_weight: bool = True):
        super(FTLearnedPositionalEmbedding, self).__init__()
        self.offset = 2
        self.num_embeddings, self.embedding_dim = weight.shape
        self.num_embeddings -= self.offset
        if save_weight:
            self.weight = weight

    def _forward(self, x: t.Tensor, **_) -> t.Tensor:
        raise NotImplementedError

    def _get_pos(self, attn_mask: t.LongTensor, past_kv_length: int = 0) -> t.Tensor:
        attention_mask = attn_mask.long()

        # create positions depending on the attention_mask
        positions = t.cumsum(attention_mask, 1).type_as(attention_mask) * attention_mask
        positions = positions.long() - 1

        # cut positions if ``past_key_values_length`` is > 0
        positions = positions[:, past_kv_length:]
        return positions + self.offset

    def forward(
        self, attention_mask: t.LongTensor, past_key_values_length: int = 0
    ) -> t.Tensor:
        pos = self._get_pos(attention_mask, past_key_values_length)
        return F.embedding(pos, self.weight).requires_grad_(True)

    @classmethod
    def from_base(cls, embedding: nn.Embedding) -> FTLearnedPositionalEmbedding:
        return cls(embedding.weight)

    def quantized_copy(self) -> QuantizedLearnedPositionalEmbedding:
        # no need to deepcopy the current layer since weight will be copied
        # when quantized.
        return QuantizedLearnedPositionalEmbedding.from_base(self)


class QuantizedLearnedPositionalEmbedding(
    QuantizedModule, FTLearnedPositionalEmbedding
):
    def __init__(self, weight: t.ByteTensor, absmax: t.FloatTensor, code: t.IntTensor):
        super(QuantizedLearnedPositionalEmbedding, self).__init__(
            weight=weight, save_weight=False
        )
        self.register_buffer("quant_weight", weight.requires_grad_(False))
        self.register_buffer("quant_absmax", absmax.requires_grad_(False))
        self.register_buffer("quant_code", code.requires_grad_(False))

    def dequantize_weight(self) -> t.Tensor:
        return dequantize_blockwise(
            self.quant_weight, absmax=self.quant_absmax, code=self.quant_code
        )

    def forward(
        self, attention_mask: t.LongTensor, past_key_values_length: int = 0
    ) -> t.Tensor:
        pos = self._get_pos(attention_mask, past_key_values_length)
        deq_weight = self.dequantize_weight()
        output = F.embedding(pos, deq_weight).float()

        if self.lora_adapter:
            return self.lora_adapter(pos) + output

        return output

    @classmethod
    def from_base(
        cls, embedding: Union[nn.Embedding, FTLearnedPositionalEmbedding]
    ) -> QuantizedLearnedPositionalEmbedding:
        weights_int8, state = quantize_blockwise_lowmem(embedding.weight)
        return cls(weights_int8, *state)


# LoRA Adapters ================================================================


class LinearAdapter(nn.Module):
    def __init__(
        self,
        in_features: int,
        out_features: int,
        r: int = 4,
        alpha: int = 1,
        dropout: float = 0.0,
        bias: bool = True,
        device: Optional[t.device] = None,
    ):
        """LoRA adapter for a linear layer"""
        super().__init__()

        self.A = nn.Parameter(t.zeros((r, in_features))).requires_grad_(True)
        nn.init.kaiming_uniform_(self.A, a=math.sqrt(5))

        self.B = nn.Parameter(t.zeros((out_features, r))).requires_grad_(True)
        nn.init.zeros_(self.B)

        if bias:
            self.bias = nn.Parameter(t.zeros((out_features,))).requires_grad_(True)
            nn.init.zeros_(self.bias)
        else:
            self.bias = None

        self.scaling = alpha / r
        self.dropout = nn.Dropout(p=dropout)

        self.to(device=device)

    def forward(self, x: t.Tensor) -> t.Tensor:
        tmp = self.dropout(x) @ self.A.T @ self.B.T
        if self.bias is not None:
            tmp += self.bias
        return tmp * self.scaling


class EmbeddingAdapter(nn.Module):
    def __init__(
        self,
        num_embeddings: int,
        embedding_dim: int,
        r: int = 4,
        alpha: int = 1,
        device: t.device = None,
    ):
        """LoRA adapter for an embedding layer"""
        super().__init__()
        self.A = nn.Parameter(t.zeros((r, num_embeddings))).requires_grad_(True)
        nn.init.zeros_(self.A)

        self.B = nn.Parameter(t.zeros((embedding_dim, r))).requires_grad_(True)
        nn.init.normal_(self.B)

        self.scaling = alpha / r
        self.to(device=device)

    def forward(self, x: t.Tensor, *args, **kwargs) -> t.Tensor:
        return (F.embedding(x, self.A.T, **kwargs) @ self.B.T) * self.scaling


# API functions ================================================================


def get_lora_adaptable_modules(
    model: nn.Module,
    target_modules: List[Union[Type[nn.Module], Type[FTModule]]] = [FTModule],
) -> Set[str]:
    """Returns a set of module names from the provided list of types onto which
    you can add LoRA adapters."""

    if not _is_quantized(model) and target_modules == [FTModule]:
        raise RuntimeWarning(
            "WARNING: model not yet quantized."
            "Results likely to be incorrect. Please speicfy target_modules manually."
        )

    module_names: set[str] = set()
    for m in model.modules():
        for n, c in m.named_children():
            if any([isinstance(c, t) for t in target_modules]):
                module_names.add(n)
    return module_names


def quantize_base_model(
    model: nn.Module, modules_not_to_quantize: Set[str] = {"lm_head"}
) -> nn.Module:
    """This function is used to quantize a large pretrained model.

    This helps to benefit from the memory savings that come from 8-bit
    quantisation of a model. You can also save this pre-quantized model to disk
    for loading in low-memory environments.

    Note, once quantized to 8-bits, there is no method implemented to
    de-quantize that layer back to FP32. If in doubt, skip over layers to not
    quantize by listing them in modules_not_to_quantize.

    The quantized 8-bit weights are frozen. The only way to fine-tune a
    quantized model is by adding on an adapter.

    Args:
        model: the model to quantize.
        modules_not_to_quantize: a list of all the module names not to quantize.

    Returns:
        nn.Module: the updated model (it is updated in place, but having this
            reference can be convenient)

    Raises:
        RuntimeError: if attempting to re-quantize an already quantized model.
    """

    if _is_quantized(model):
        raise RuntimeError("Attempt to re-quantize an already quantized model")

    for name, mod in model.named_children():

        # Recurse:
        if len(list(mod.children())) > 0:
            quantize_base_model(mod, modules_not_to_quantize)

        # Replace modules:
        if isinstance(mod, nn.Linear):
            if name in modules_not_to_quantize:
                model._modules[name] = FTLinear.from_linear(mod)
            else:
                model._modules[name] = QuantizedLinear.from_linear(mod)

        elif "LearnedPositionalEmbedding" in str(type(mod)):
            # For OPT, the learned positional embedding works differently to
            # the vanilla Embedding, so we need a special class for this:
            assert isinstance(mod, nn.Embedding)
            if name in modules_not_to_quantize:
                model._modules[name] = FTLearnedPositionalEmbedding.from_base(mod)
            else:
                model._modules[name] = QuantizedLearnedPositionalEmbedding.from_base(
                    mod
                )

        elif isinstance(mod, nn.Embedding):
            if name in modules_not_to_quantize:
                model._modules[name] = FTEmbedding.from_embedding(mod)
            else:
                model._modules[name] = QuantizedEmbedding.from_embedding(mod)
        else:
            # For any other module in the pretrained model, freeze weights.
            model._modules[name] = model._modules[name].requires_grad_(False)

    model.register_buffer("_is_quantized", t.tensor([True]))

    return model


def _validate_new_ft_args(
    model: nn.Module,
    adapt_layers: Optional[Set[str]],
    plain_layers: Set[str],
    modules_not_to_freeze: Set[str],
    modules_not_to_quantize: Set[str],
) -> Tuple[Set[str], Set[str]]:
    """Validates args for new_finetuned.
    Note, sets are passed by reference."""

    # does not support lora adapting plain layers

    if adapt_layers is None:
        # TODO: write tests to ensure that these are the same in both cases
        if _is_quantized(model):
            adapt_layers = get_lora_adaptable_modules(model)
        else:
            adapt_layers = get_lora_adaptable_modules(
                model, target_modules=[nn.Linear, nn.Embedding]
            )
        for p in plain_layers:
            adapt_layers.remove(p)

    for m in plain_layers:
        if m in adapt_layers:
            raise ValueError(
                f"{m} cannot feature in both plain_layers and adapt_layers"
            )
        modules_not_to_freeze.add(m)
        modules_not_to_quantize.add(m)

    quantized_modules = adapt_layers - modules_not_to_quantize

    for m in modules_not_to_freeze:
        if m in adapt_layers:
            raise ValueError(f"{m} cannot both be un-frozen and have an adapter.")

    return adapt_layers, quantized_modules


def new_finetuned(
    model: nn.Module,
    adapt_layers: Optional[Set[str]] = None,
    plain_layers: Set[str] = {"lm_head"},
    # fmt: off
    embedding_config: Union[Dict[str, EmbeddingAdapterConfig], EmbeddingAdapterConfig] = EmbeddingAdapterConfig(),
    linear_config: Union[Dict[str, LinearAdapterConfig], LinearAdapterConfig] = LinearAdapterConfig(),
    # fmt: on
    modules_not_to_quantize: Set[str] = set(),
    modules_not_to_freeze: Set[str] = set(),
) -> nn.Module:
    """Create a new finetuned model from a pretrained model whose weights will
    be shared.

    Layers can either be:
    - quantized to 8-bits or not
    and
    - frozen, not-frozen or adapted

    This gives 6 combinations. By default, we want to quantize everything we
    can, and add adapters to everything. However, when deviating from this
    default, there are two main types of layers we care about when fine-tuning:

    1. The first are layers we want to add LoRA adapters to, such as
    query and value projection matrices in the attention module. That is, the
    pretrained weights are quantized and frozen, and we train low-rank, FP32
    'adapters'. You can choose which layers we treat like this by naming them
    in the ``adapt_layers`` argument,

    2. The second are layers that we just want to leave alone, such as the
    language model head. By 'leave alone', we mean that these are vanilla NN
    layers: not quantized and not-frozen (nor adapted). You can list the layers
    to treat as such in the ``plain_layers`` argument. These cannot intersect
    with ``adapt_layers``.

    To give full control over what happens to each layer, we also provide the
    modules_not_to_freeze and modules_not_to_quantize arguments.

    See README for more info.

    Args:
        model: base pretrained model. Can be quantized already.
        target_layers: the targets onto which to add LoRA adapters. Only Linear
            and Embedding layers are suitable for this. If omitted, all
            suitable layers are adapted.
        plain_layers: layers not to quantize or freeze (or LoRA-adapt).
        embedding_config: the configuration to use for the added LoRA
            EmbeddingAdapters. Either a single config to apply to all layers, or a
            dict of configs for each embedding layer in target_layers.
        linear_config: the configuration to use for the added LoRA
            LinearAdapters. Either a single config to apply to all layers, or a
            dict of configs for each linear layer in target_layers.
        modules_not_to_quantize: don't quantize this layer. Error if already
            quantized in previous quantize_base_model() call.
            WARNING: not thorouoghly tested.
        modules_not_to_freeze: don't freeze this layer's weights. Error if
            adapted (i.e. base weights are not frozen, and there is an adaptor)
            WARNING: not thorouoghly tested.

    Raises:
        ValueError: For not exhaustively specifying the adapter configs when
            using a dict in either ``embedding_config`` or ``linear_config``.
    """

    adapt_layers, quantized_modules = _validate_new_ft_args(
        model,
        adapt_layers,
        plain_layers,
        modules_not_to_freeze,
        modules_not_to_quantize,
    )

    if not _is_quantized(model):
        # Here, we run quantize_base_model but with modules_not_to_quantize set
        # to everything, thus allowing us to place LoRA adapters on all
        # compatible layers.

        mntq = get_lora_adaptable_modules(
            model, target_modules=[nn.Linear, nn.Embedding]
        )
        quantize_base_model(model, mntq)

    new_model = copy_mod(model)

    for module in list(new_model.modules()):
        for name, child in module.named_children():

            # Linear Modules -------------------------------------------------------

            if isinstance(child, FTLinear):  # QuantizedLinear or FTLinear

                if name in quantized_modules and not isinstance(child, QuantizedModule):
                    module._modules[name] = child.quantized_copy()

                if name in adapt_layers:
                    config = _get_config(linear_config, name, ConfigType.linear)
                    module._modules[name].requires_grad_(False)
                    module._modules[name].lora_adapter = LinearAdapter(
                        child.in_features,
                        child.out_features,
                        r=config.r,
                        alpha=config.alpha,
                        dropout=config.dropout,
                        bias=config.bias,
                        device=_get_ft_mod_device(child),
                    ).requires_grad_(True)

                if name in modules_not_to_freeze:
                    module._modules[name] = plain_copy(child)

            # Embedding Modules ---------------------------------------------------

            elif isinstance(child, FTEmbedding) or isinstance(
                child, FTLearnedPositionalEmbedding
            ):
                if name in quantized_modules and not isinstance(child, QuantizedModule):
                    module._modules[name] = child.quantized_copy()

                if name in adapt_layers:
                    config = _get_config(embedding_config, name, ConfigType.embedding)
                    module._modules[name].requires_grad_(False)
                    module._modules[name].lora_adapter = EmbeddingAdapter(
                        num_embeddings=child.num_embeddings,
                        embedding_dim=child.embedding_dim,
                        r=config.r,
                        alpha=config.alpha,
                        device=_get_ft_mod_device(child),
                    ).requires_grad_(True)

                if name in modules_not_to_freeze:
                    module._modules[name] = plain_copy(child)

            # Unknown modules -----------------------------------------------------

            elif name in adapt_layers:
                print(f"WARNING: cannot adapt {name} of type {child._get_name()}")

            elif name in modules_not_to_freeze:
                # Fine-tune arbitrary (i.e. not Linear or Embedding) layers.
                # - these cannot be shared between different fine-tuned models
                # - must require_grad.
                module._modules[name] = copy.deepcopy(child).requires_grad_(True)

    return new_model
