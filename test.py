from transformers import AutoTokenizer, AutoModel
import torch

tokenizer = AutoTokenizer.from_pretrained("openai-community/gpt2")
model = AutoModel.from_pretrained("openai-community/gpt2")

inputs = tokenizer("Hello world", return_tensors="pt")

def shape_only(name):
    def hook(_, __, out):
        # 只取张量并打印 shape
        shapes = [tuple(o.shape) for o in (out if isinstance(out, tuple) else (out,)) if torch.is_tensor(o)]
        print(f"{name:30} -> {shapes}")
    return hook

for n, m in model.named_modules():
    m.register_forward_hook(shape_only(n))

with torch.no_grad():
    model(**inputs)