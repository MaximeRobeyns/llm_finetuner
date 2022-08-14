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
Then clone this repo, navigate to the root and use:
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

# You can manually quantize a pre-trained model and save it so you don't have
# to re-quantize it each time:
ft.quantize_base_model(base_model)

# Create new finetuned models using either the base or quantized model
model_1 = ft.new_finetuned(base_model)
# Can specify granular parameters, if required
model_2 = ft.new_finetuned(
    base_model,
    target_layers = {'embed_tokens', 'embed_posotions', 'q_proj', 'v_proj'},
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
