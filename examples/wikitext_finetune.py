import torch as t
import torch.nn.functional as F
import finetuna as ft
import transformers

from tqdm import tqdm
from datasets import load_dataset
from torch.optim import Adam

device = "cuda:0" if t.cuda.is_available() else "cpu"
model_name = "facebook/opt-125m"

# NOTE: because the weights are shared with later models, both have to be
# placed on the same device.
model = transformers.AutoModelForCausalLM.from_pretrained(model_name).to(device)

tokenizer = transformers.AutoTokenizer.from_pretrained(model_name)
tokenizer.pad_token = model.config.eos_token_id

model_1 = ft.new_finetuned(
    model,
    target_layers={"embed_tokens", "embed_posotions", "q_proj", "v_proj"},
).to(device)
model_2 = ft.new_finetuned(
    model,
    target_layers={"embed_tokens", "embed_posotions", "q_proj", "v_proj"},
).to(device)
model_1.gradient_checkpointing_enable()

dset = load_dataset("wikitext", "wikitext-2-raw-v1", split="train")

optimizer = Adam(model_1.parameters(), lr=1e-5)

batch_size = 20

with t.cuda.amp.autocast():
    # trange = tqdm(range(0, 200, batch_size))
    trange = tqdm(range(0, len(dset), batch_size))
    for i in trange:
        rows = [r for r in dset[i : i + batch_size]["text"] if len(r) > 1]

        batch = tokenizer(
            rows, truncation=True, padding=True, max_length=128, return_tensors="pt"
        )
        batch = {k: v.to(device) for k, v in batch.items()}

        out = model_1.forward(
            **batch,
        )

        loss = F.cross_entropy(
            out.logits[:, :-1, :].flatten(0, -2),
            batch["input_ids"][:, 1:].flatten(),
            reduction="mean",
        )

        # print(loss)
        trange.set_description(f"Loss {loss:.4f}")
        loss.backward()
        optimizer.step()
        optimizer.zero_grad()
