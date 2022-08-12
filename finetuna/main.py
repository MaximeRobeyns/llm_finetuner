import copy
import math
import torch as t
import torch.nn as nn
import torch.nn.functional as F

from abc import abstractmethod
from typing import Type, Optional, Union
from dataclasses import dataclass
from torch.cuda.amp import custom_fwd, custom_bwd
from bitsandbytes.functional import quantize_blockwise, dequantize_blockwise
from bitsandbytes.nn.modules import StableEmbedding


@dataclass
class EmbeddingAdapterConfig:
    r: int = 4
    alpha: int = 1


@dataclass
class LinearAdapterConfig:
    r: int = 4
    alpha: int = 1
    dropout: float = 0.0


class FTModule(nn.Module):
    def __init__(self, weight: t.ByteTensor, absmax: t.FloatTensor, code: t.IntTensor):
        """A frozen, 8-bit quantized module which accomodates LoRA adapters.

        Args:
            weight: the blockwize-quantized 8-bit weights from the underlying
                module.
            absmax: the maximum block values of the weight tensor
            code: the quantization map
        """
        super().__init__()
        self.register_buffer("quant_weight", weight.requires_grad_(False))
        self.register_buffer("quant_absmax", absmax.requires_grad_(False))
        self.register_buffer("quant_code", code.requires_grad_(False))

        self.lora_adapter: Optional[nn.Module] = None

    @abstractmethod
    def _forward(self, x: t.Tensor, **kwargs) -> t.Tensor:
        """This method must implement the forward pass for the module being
        replaced, in which the frozen weights are dequantized before being used
        in the layer and then discarded.
        """
        raise NotImplementedError

    def forward(self, *args, **kwargs) -> t.Tensor:
        """Adds adapter output to normal layer output, if a LoRA adapter has
        been set for this layer."""
        output = self._forward(*args, **kwargs)
        if self.lora_adapter:
            return self.lora_adapter(args[0]) + output
        return output


class DequantizeAndLinear(t.autograd.Function):
    """
    Wrapper around `nn.Linear` to dequantize the weight matrices as required.
    """

    @staticmethod
    @custom_fwd
    def forward(
        ctx,
        input: t.Tensor,
        weights_quantized: t.ByteTensor,
        absmax: t.FloatTensor,
        code: t.FloatTensor,
        bias: t.FloatTensor,
    ) -> t.Tensor:
        """Block-dequantize the 8-bit weights to use them in the forward pass.

        Args:
            input: the tensor to pass through the underlying nn.Linear function
            weights_quantized: the stored 8-bit weight tensor to dequantize
            absmax: the block's absolute maximum for dequantization
            code: the dynamic quantization map (maps uint8 indices to float32
                values, see 1511.04561)
            bias: bias term (not quantized)
        Returns:
            t.Tensor: the input passed through the linear layer.
        """

        weights_deq = dequantize_blockwise(weights_quantized, absmax=absmax, code=code)
        assert (
            weights_deq.requires_grad == False
        ), "Frozen weights have requires_grad = True"
        ctx.save_for_backward(input, weights_quantized, absmax, code)
        ctx._has_bias = bias is not None
        return F.linear(input, weights_deq, bias)

    @staticmethod
    @custom_bwd
    def backward(ctx, grad_output: t.Tensor):
        """Block-dequantize the 8-bit weights to update the gradient in the
        backward pass.

        Recall that the weights are frozen (i.e. a constant in dL/dW), so we
        merely multiply the current gradient value by the dequantized weights.

        Args:
            grad_output: the current gradient value in the backward pass
        """
        assert (
            not ctx.needs_input_grad[1]
            and not ctx.needs_input_grad[2]
            and not ctx.needs_input_grad[3]
        )
        # TODO: can we get rid of input?
        input, weights_quantized, absmax, code = ctx.saved_tensors
        # grad output has shape: [*batch, out_features]
        weights_deq = dequantize_blockwise(weights_quantized, absmax=absmax, code=code)
        grad_input = grad_output @ weights_deq
        grad_bias = grad_output.flatten(0, -2).sum(dim=0) if ctx._has_bias else None
        return grad_input, None, None, None, grad_bias


class QuantizedLinear(FTModule):
    def __init__(
        self,
        weight: t.ByteTensor,
        absmax: t.FloatTensor,
        code: t.IntTensor,
        bias: Optional[nn.Parameter] = None,
    ):
        super().__init__(weight, absmax, code)
        assert isinstance(bias, nn.Parameter) or bias is None
        self.out_features, self.in_features = weight.shape
        self.bias = bias

    def _forward(self, x: t.Tensor, **kwargs) -> t.Tensor:
        return DequantizeAndLinear.apply(
            x,
            self.quant_weight,
            self.quant_absmax,
            self.quant_code,
            self.bias,
        )

    @classmethod
    def from_linear(cls, linear: nn.Linear) -> "QuantizedLinear":
        """Converts a PyTorch nn.Linear module to a quantized 8-bit linear
        layer with frozen body parameters"""
        weights_int8, state = quantize_blockwise_lowmem(linear.weight)
        return cls(weights_int8, *state, linear.bias)

    def __repr__(self):
        return f"{self._get_name()} ({self.in_features}, {self.out_features})"


class QuantizedEmbedding(FTModule):
    def __init__(self, weight: t.ByteTensor, absmax: t.FloatTensor, code: t.IntTensor):
        """A frozen, quantized Embedding layer which accomodates LoRA adapters

        Args:
            weight: the blockwise-quantized 8-bit weights from the underlying
                nn.Embedding module.
            absmax: the maximum block values of the weight tensor
            code: the quantization map
        """
        super().__init__(weight, absmax, code)
        self.num_embeddings, self.embedding_dim = weight.shape

    def _forward(self, x: t.Tensor, **kwargs):
        with t.no_grad():
            # Note: both the quantized weights and input indices are not differentiable
            weight_deq = dequantize_blockwise(
                self.quant_weight, absmax=self.quant_absmax, code=self.quant_code
            )
            # TODO: use stable embedding
            # https://bitsandbytes.readthedocs.io/en/latest/tree/bitsandbytes.nn.html#module-bitsandbytes.nn.modules
            return F.embedding(x, weight_deq, **kwargs)

    @classmethod
    def from_embedding(cls, embedding: nn.Embedding) -> "QuantizedEmbedding":
        """Converts a PyTorch nn.Embedding module to a quantized 8-bit version
        with frozen values."""
        weights_int8, state = quantize_blockwise_lowmem(embedding.weight)
        return cls(weights_int8, *state)

    def __repr__(self):
        return f"{self._get_name()} ({self.num_embeddings}, {self.embedding_dim})"


class QuantizedLearnedPositionalEmbedding(FTModule):
    def __init__(self, weight: t.ByteTensor, absmax: t.FloatTensor, code: t.IntTensor):
        super().__init__(weight, absmax, code)
        self.offset = 2
        self.num_embeddings, self.embedding_dim = weight.shape
        self.num_embeddings -= self.offset

    def forward(
        self, attention_mask: t.LongTensor, past_key_values_length: int = 0
    ) -> t.Tensor:
        attention_mask = attention_mask.long()

        # create positions depending on the attention_mask
        positions = t.cumsum(attention_mask, 1).type_as(attention_mask) * attention_mask
        positions = positions.long() - 1

        # cut positions if ``past_key_values_length`` is > 0
        positions = positions[:, past_key_values_length:]
        positions = positions + self.offset

        with t.no_grad():
            # both the weights and input indices are not differentiable:
            weight_deq = dequantize_blockwise(
                self.quant_weight, absmax=self.quant_absmax, code=self.quant_code
            )
            output = F.embedding(positions, weight_deq)

        if self.lora_adapter:
            return self.lora_adapter(positions) + output

        return output

    @classmethod
    def from_base(
        cls, embedding: nn.Embedding
    ) -> "QuantizedLearnedPositionalEmbedding":
        weights_int8, state = quantize_blockwise_lowmem(embedding.weight)
        return cls(weights_int8, *state)


def _is_quantized(model: nn.Module) -> bool:
    try:
        q = model._is_quantized.item()
        assert isinstance(q, bool)
        return q
    except AttributeError:
        return False


#def get_lora_adaptable_modules(
#    model: nn.Module,
#    target_modules: list[Union[Type[nn.Module], Type[FTModule]]] = [FTModule],
#) -> set[str]:
def get_lora_adaptable_modules(
    model: nn.Module,
    target_modules: list = [FTModule],
) -> set:
    """Returns a set of module names from the provided list of types onto which
    you can add LoRA adapters."""
    if not _is_quantized(model):
        print("WARNING: model is not yet quantized")

    module_names: set[str] = set()
    for m in model.modules():
        for n, c in m.named_children():
            if any([isinstance(c, t) for t in target_modules]):
                module_names.add(n)
    return module_names


def get_model_mem_consumption(model) -> int:
    """Returns an estimate of the model's memory consumption by looking at its
    parameters."""
    mem_params = sum(
        [param.nelement() * param.element_size() for param in model.parameters()]
    )
    mem_bufs = sum([buf.nelement() * buf.element_size() for buf in model.buffers()])
    return mem_params + mem_bufs


# Quantization  ---------------------------------------------------------------


def quantize_blockwise_lowmem(matrix: t.Tensor, chunk_size: int = 2**20):
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

    try:
        flat_tensor = matrix.data.to(dtype=t.float32).view(-1)
    except Exception:
        flat_tensor = matrix.data.to(dtype=t.float32).contiguous().view(-1)

    for i in range((matrix.numel() - 1) // chunk_size + 1):
        input_chunk = flat_tensor[i * chunk_size : (i + 1) * chunk_size].clone()
        quantized_chunk, (absmax_chunk, code) = quantize_blockwise(
            input_chunk, code=code
        )
        chunks.append(quantized_chunk)
        absmaxes.append(absmax_chunk)

    matrix_i8 = t.cat(chunks).reshape_as(matrix)
    absmax = t.cat(absmaxes)
    return matrix_i8, (absmax, code)


# layers -----------------------------------------------------------------------


class LinearAdapter(nn.Module):
    def __init__(
        self,
        body_weight: t.Tensor,
        r: int = 4,
        alpha: int = 1,
        dropout: float = 0.0,
        device: t.device = None,
    ):
        """LoRA adapter for a linear layer"""
        super().__init__()
        out_features, in_features = body_weight.shape
        self.lora_A = nn.Parameter(t.zeros((r, in_features))).requires_grad_(True)
        nn.init.kaiming_uniform_(self.lora_A, a=math.sqrt(5))
        self.lora_B = nn.Parameter(t.zeros((out_features, r))).requires_grad_(True)

        nn.init.zeros_(self.lora_B)
        self.scaling = alpha / r
        self.dropout = nn.Dropout(p=dropout)

        if device is None:
            device = body_weight.device
        self.to(device=device)

    def forward(self, x: t.Tensor) -> t.Tensor:
        return (self.dropout(x) @ self.lora_A.T @ self.lora_B.T) * self.scaling


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
        self.lora_A = nn.Parameter(t.zeros((r, num_embeddings))).requires_grad_(True)
        nn.init.zeros_(self.lora_A)
        self.lora_B = nn.Parameter(t.zeros((embedding_dim, r))).requires_grad_(True)
        nn.init.normal_(self.lora_B)
        self.scaling = alpha / r

        self.to(device=device)

    def forward(self, x: t.Tensor, *args, **kwargs) -> t.Tensor:
        return (F.embedding(x, self.lora_A.T, **kwargs) @ self.lora_B.T) * self.scaling


def quantize_base_model(model: nn.Module):
    """Modifies the provided module in place to quantize it. (No shallow copying etc.)"""
    for module in list(model.modules()):
        for name, child in module.named_children():
            if type(child) is nn.Linear:
                setattr(module, name, QuantizedLinear.from_linear(child))
            elif "LearnedPositionalEmbedding" in str(type(child)):
                # For OPT, the learned positional embedding works differently to
                # the vanilla Embedding, so we need a special class for this:
                assert isinstance(child, nn.Embedding)
                setattr(
                    module, name, QuantizedLearnedPositionalEmbedding.from_base(child)
                )
            elif type(child) is nn.Embedding:
                setattr(module, name, QuantizedEmbedding.from_embedding(child))
            else:
                # remove gradients
                child.requires_grad_(False)

    model.register_buffer("_is_quantized", t.tensor([True]))


def copy_mod(mod: nn.Module) -> nn.Module:
    tmp = copy.copy(mod)
    tmp._parameters = copy.copy(mod._parameters)
    tmp._buffers = copy.copy(mod._buffers)
    tmp._modules = {name: copy_mod(child) for (name, child) in mod._modules.items()}
    return tmp


#def new_finetuned(
#    model: nn.Module,
#    target_layers: Optional[set[str]] = None,
#    # fmt: off
#        embedding_config: Union[dict[str, EmbeddingAdapterConfig], EmbeddingAdapterConfig] = EmbeddingAdapterConfig(),
#        linear_config: Union[dict[str, LinearAdapterConfig], LinearAdapterConfig] = LinearAdapterConfig(),
#    # fmt: on
#) -> nn.Module:
def new_finetuned(
    model: nn.Module,
    target_layers: Optional = None,
    # fmt: off
        embedding_config: Union = EmbeddingAdapterConfig(),
        linear_config: Union = LinearAdapterConfig(),
    # fmt: on
) -> nn.Module:
    """
    Shallow copies and quantizes if necessary
    """

    if not _is_quantized(model):
        quantize_base_model(model)

    new_model = copy_mod(model)

    if target_layers is None:
        target_layers = get_lora_adaptable_modules(new_model)

    for n, module in new_model.named_modules():
        root_name = n.split(".")[-1]

        if root_name in target_layers:

            # Linear adapter --------------------------------------------------
            if isinstance(module, QuantizedLinear):
                if isinstance(linear_config, dict):
                    config = linear_config.get(root_name)
                    if not config:
                        raise ValueError(
                            f"No LinearAdapterConfig provided for layer: {root_name}"
                        )
                else:
                    config = linear_config
                # the weight is merely used to get the dimensions
                weight = module.get_buffer("quant_weight")
                setattr(
                    module,
                    "lora_adapter",
                    LinearAdapter(
                        body_weight=weight,
                        r=config.r,
                        alpha=config.alpha,
                        dropout=config.dropout,
                        device=weight.device,
                    ),
                )

            # Embedding adapter ------------------------------------------------
            elif isinstance(module, QuantizedEmbedding) or isinstance(
                module, QuantizedLearnedPositionalEmbedding
            ):
                if isinstance(embedding_config, dict):
                    config = embedding_config.get(root_name)
                    if not config:
                        raise ValueError(
                            f"No EmbeddingAdapterConfig provided for layer: {root_name}"
                        )
                else:
                    config = embedding_config
                # print(f"adding embedding adapter to {n}")
                weight = module.get_buffer("quant_weight")
                module.lora_adapter = EmbeddingAdapter(
                    num_embeddings=module.num_embeddings,
                    embedding_dim=module.embedding_dim,
                    r=config.r,
                    alpha=config.alpha,
                    device=weight.device,
                )

            else:
                print(f"WARNING: cannot adapt {n} of type {module._get_name()}")

    return new_model
