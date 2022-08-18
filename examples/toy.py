import io
import copy
import math
import torch as t
import torch.nn as nn
import torch.nn.functional as F
import warnings
import accelerate
import contextlib

with warnings.catch_warnings():
    warnings.simplefilter("ignore")
    with contextlib.redirect_stdout(io.StringIO()):
        from bitsandbytes.functional import quantize_blockwise, dequantize_blockwise
        from bitsandbytes.nn import Linear8bitLt, Int8Params

weight = t.nn.Parameter(t.randn(2, 5))
bias = t.nn.Parameter(t.randn(2))

vanilla = nn.Linear(5, 2, bias=True)
vanilla.weight = weight
vanilla.bias = bias
vanilla = vanilla.to(device=0, dtype=t.float16)

input_ = t.randn(1, 5, dtype=t.float16).to(0)
with t.cuda.amp.autocast():
    print(vanilla(input_))

sd = vanilla.state_dict()

# with accelerate.init_empty_weights():
linear = Linear8bitLt(5, 2, bias=True, has_fp16_weights=False)
# linear.load_state_dict(sd)
linear.load_state_dict({"weight": weight}, strict=False)
linear.load_state_dict({"bias": bias}, strict=False)
# linear.load_state_dict({"weight": weight, "bias": None})
# linear.weight = Int8Params(weight.data)
# linear.bias = Int8Params(bias.data)
linear = linear.to(0)

print(linear(input_))

# for m in model.modules():
#     for n, c in m.named_children():
#         print(f"\n{n}")
#         # print(f"{n}: {c}")
#         # print(m._modules[n])
#         for nn, p in m._modules[n].named_parameters():
#             print(f"{nn}: {p}, {p.dtype}")
