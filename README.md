<h1 align="center">LLM Fine Tuner</h1>
<p align="center">
  <img src="./docs/source/_static/fish.jpg">
</p>

Fine-tune multiple large language models in low-memory environments.

This repository provides wrappers around LLMs for
- [8-bit quantization](https://arxiv.org/pdf/2110.02861.pdf) of pre-trained model weights, and
- fine-tuning with [LoRA adapters](https://arxiv.org/pdf/2106.09685.pdf).

## Install

First, you need PyTorch installed with `cudatoolkit` (see [link](https://docs.google.com/document/d/1JxSo4lQgMDBdnd19VBEoaG-mMfQupQ3XvOrgmRAVtpU/edit#)).  You don't seem to get `cudatoolkit` when you install PyTorch using `pip`.  You may be able to install `cudatoolkit` separately, but we haven't tested that.  Our tested proceedure is to remove previous installs of PyTorch.  e.g. if PyTorch is currently installed through pip:
```bash
pip uninstall torch
```
Then, install PyTorch with `cudatookkit` through conda (see https://pytorch.org for the latest install command, but CUDA 11.3 seems to be working with `bitsandbytes`)
```bash
conda install pytorch torchvision torchaudio cudatoolkit=11.3 -c pytorch
```
Then clone this repo, navigate to the repository root and use:
```bash
pip install -e .
```

## Notes on bitsandbytes
`bitsandbytes` was forked in a confusing way.
The actual repo is https://github.com/TimDettmers/bitsandbytes, and this version installs with `pip install bitsandbytes`.  However, if you search on Google, you get the older repo https://github.com/facebookresearch/bitsandbytes and this version installs with `pip install bitsandbytes-cudaXXX`.  We need the newer version, installed with `pip install bitsandbytes`.

## Example

```python
import finetuna as ft
import transformers

model_name = 'facebook/opt-125m'
base_model = transformers.AutoModelForCausalLM.from_pretrained(model_name)

# If memory constraints require it, you can manually pre-quantize a model:
ft.quantize_base_model(base_model)

# Create new finetuned models using either the base or quantized model
model_1 = ft.new_finetuned(base_model)

# Can specify granular parameters, if required
model_2 = ft.new_finetuned(
    base_model,
    adapt_layers = {'embed_tokens', 'embed_posotions', 'q_proj', 'v_proj'},
    embedding_config=ft.EmbeddingAdapterConfig(r=4, alpha=1),
    linear_config={
        'q_proj': ft.LinearAdapterConfig(r=8, alpha=1, dropout=0.0, bias=False),
        'v_proj': ft.LinearAdapterConfig(r=4, alpha=1, dropout=0.1, bias=True),
    },
)


# Fine-tune as usual
with t.cuda.amp.autocast():
    opt.zero_grad()
    loss = mse_loss(model_1(prompt) - target)  # pseudo-notation
    model_1.backward(loss)
    opt.step()


# NOTE: saving not yet implemented:

# Either save complete state like a normal pytorch model
t.save(model_1.state_dict(), "/save/path.pt")

# Or save only the changed state to reload from base model
t.save(ft.state_dict(model_1), "/save/path_finetuned.pt")

# Load:
model_2.load_state_dict("/save/path.pt")

# Load only adapter state with strict=False
model_2.load_state_dict("/save/path_finetuned.pt", strict=False)
```

## Usage

### Optional Explicit Quantization

If you have memory constraints which mean that you can't keep a full model
loaded into memory, then use the ``quantize_base_model(model)`` function to
convert all compatible (i.e. nn.Linear and nn.Embedding) modules into frozen,
8-bit versions.

```python
base_model = transformers.AutoModelForCausalLM.from_pretrained(model_name)
ft.quantize_base_model(base_model)
```

Note that if you quantize a layer in ``quantize_base_model``, you cannot then
require that it is no longer quantized in ``new_finetuned``. Later versions of
this library may support this, but issue a warning owing to the loss of
accuracy incurred by the round trip from FP32 -> INT8 -> FP32.

If you think that you might need an un-quantized version of a layer, add it to
the ``modules_not_to_quantize`` argument when callin ``quantize_base_model``.
Un-quantized layers will be automatically quantized during ``new_finetuned`` if
needed (although their un-quantized, pre-trained weights will stay in memory as
part of the base model).

### Creating New Finetuned Models

Once you have a base model, you will create some new models to fine-tune using
the ``new_finetuned`` function. The most basic invocation without any arguments
will:

- 'freeze' and quantize all the pretrained Embedding and Linear modules (except 'lm_head')
- add LoRA adapters to all Embedding and Linear modules (except 'lm_head'), using default adapter configs
- 'freeze' all other nn.Module parameters

```python
ft1 = ft.new_finetuned(base_model)
```

#### Using the ``adapt_layers`` argument

If you wish to only adapt certain layers, then you can specify these layers in
the ``adapt_layers`` argument:

```python
ft2 = ft.new_finetuned(base_model, adapt_layers={"q_proj", "v_proj"})
```
This means that we:
- 'freeze' and quantize all the pretrained Embedding and Linear modules (except 'lm_head')
- add LoRA adapters to q_proj and v_proj only
- 'freeze' all other nn.Module parameters

#### Using the ``plain_layers`` argument

Sometimes we want to keep a layer the same as in the base model, and fine-tune
it directly.

An example is the 'lm_head' module from the previous examples: it is not frozen
nor quantized, so when we do gradient steps, we can update its parameters as
usual.

You can specify nn.Linear layers, nn.Embedding layers or any arbitrary
nn.Module in the ``plain_layers`` argument when creating a new finetuned model:

```python
ft2 = ft.new_finetuned(
    base_model,
    adapt_layers={"q_proj", "v_proj"},
    plain_layers={"lm_head", "out_proj", "layer_norm"}
)
```
This means that we:
- 'freeze' and quantize all the pretrained Embedding and Linear modules (except 'lm_head', 'out_proj' and 'layer_norm')
- add LoRA adapters to q_proj and v_proj only
- 'freeze' all other nn.Module parameters, except 'lm_head', 'out_proj' and 'layer_norm'

(TODO: document the separate use of the ``modules_not_to_quantize`` and
``modules_not_to_freeze`` arguments for more granular control.)

