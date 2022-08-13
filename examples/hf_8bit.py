import torch as t
from transformers import AutoModelForCausalLM, AutoTokenizer, AutoConfig

MAX_NEW_TOKENS = 256
# model_name = "ElutherAI/opt-125m"
model_name = "EleutherAI/gpt-j-6B"
text = """
Q: On average Joe throws 25 punches per minute. A fight lasts 5 rounds of 3 minutes.
How many punches did he throw?\n
A: Let's think step by step.\nStep 1:"""

tokenizer = AutoTokenizer.from_pretrained(model_name, use_fast=False)
input_ids = tokenizer(text, return_tensors="pt").input_ids

free_in_GB = int(t.cuda.mem_get_info()[0] / 1024**3)
max_memory = f"{free_in_GB-2}GB"
n_gpus = t.cuda.device_count()
max_memory = {i: max_memory for i in range(n_gpus)}
print(max_memory)

model = AutoModelForCausalLM.from_pretrained(
    model_name, device_map="auto", load_in_8bit=True, max_memory=max_memory
)
generated_ids = model.generate(
    input_ids,
    do_sample=True,
    num_return_sequences=1,
    max_length=MAX_NEW_TOKENS,
    return_dict_in_generate=False,
)

str_outputs = tokenizer.batch_decode(generated_ids, ignore_special_tokens=True)

for s in str_outputs:
    print(s)
