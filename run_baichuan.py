from transformers import AutoModelForCausalLM, AutoTokenizer
import torch

model_path = "/mnt/data/Baichuan2-7B-Chat"

tokenizer = AutoTokenizer.from_pretrained(
    model_path,
    trust_remote_code=True
)

model = AutoModelForCausalLM.from_pretrained(
    model_path,
    trust_remote_code=True,
    low_cpu_mem_usage=True,
    torch_dtype=torch.float16
)

model.eval()

prompt = "明明明明明白白白喜欢他，可她就是不说。 这句话里，明明和白白谁喜欢谁？"

inputs = tokenizer(prompt, return_tensors="pt")

outputs = model.generate(
    **inputs,
    max_new_tokens=128,
    do_sample=False
)

response = tokenizer.decode(
    outputs[0],
    skip_special_tokens=True
)

print("\n模型回答：")
print(response)