<h1 align="center">LLM Fine Tuner</h1>
<p align="center">
  <img src="./docs/source/_static/logo.png">
</p>

Fine-tune multiple large language models in low-memory environments.

This repository provides wrappers around LLMs for
- [8-bit quantization](https://arxiv.org/pdf/2110.02861.pdf) of pre-trained model weights, and
- fine-tuning with [LoRA adapters](https://arxiv.org/pdf/2106.09685.pdf).

## Install

# choices: {cuda92, cuda 100, cuda101, cuda102, cuda110, cuda111, cuda113}
# replace XXX with the respective number
pip install bitsandbytes-cudaXXX
pip install llm_finetuner

## Example

```python
import finetuna as ft
import transformers

model_name = 'facebook/opt-125m'
base_model = transformers.AutoModelForCausalLM(mode_name)

# convenience to pre-quantize weights so you don't have to do it each time you
# load a large pretrained model
quant_model = ft.only_quantize(base_model)

# Create new finetuned models using either the base or quantized model
# Can specify granular parameters, if required
model_1 = ft.new_finetuned(base_model)
model_2 = ft.new_finetuned(
    quant_model,
    target_layers = {'embed_tokens', 'embed_posotions', 'q_proj', 'v_proj'},
    embedding_config=ft.EmbeddingAdapterConfig(r=4, alpha=1),
    linear_config={
        'q_proj': ft.LinearAdapterConfig(r=8, alpha=1, dropout=0.0),
        'v_proj': ft.LinearAdapterConfig(r=4, alpha=1, dropout=0.1),
    },
)

# Fine-tune as usual
opt.zero_grad()
loss = mse_loss(model_1(prompt) - target)  # pseudo-notation
model_1.backward(loss)
opt.step()

# Either save complete state like a normal pytorch model
t.save(model_1.state_dict(), "/save/path.pt")

# Or save only the changed state to reload from base model
t.save(ft.state_dict(model_1), "/save/path_finetuned.pt")

# Load:
model_2.load_state_dict("/save/path.pt")
# Load only adapter state with strict=False
model_2.load_state_dict("/save/path_finetuned.pt", strict=False)
```
