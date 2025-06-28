import os
import torch
from transformers import GPT2LMHeadModel
model_config = "deps/gpt2"

model = GPT2LMHeadModel.from_pretrained(model_config).eval()

state_dict = model.state_dict()
new_state_dict = dict()
for name, param in state_dict.items():
    name = name.replace("attn.c_attn", "c_attn.fn.0")
    name = name.replace("attn.c_proj", "c_proj.fn.0")
    name = name.replace("mlp", "mlp.fn.0")
    name = name.replace("ln_f", "ln_f.fn.0")
    name = name.replace("ln_1", "ln_1.fn.0")
    name = name.replace("ln_2", "ln_2.fn.0")
    new_state_dict[name] = param

os.makedirs('deps/mot-gpt2', exist_ok=True)
torch.save(new_state_dict, 'deps/mot-gpt2/model_state_dict.pth')
