from transformers import AutoTokenizer, AutoModel
import torch

model_path = "/mnt/data/chatglm3-6b"

tokenizer = AutoTokenizer.from_pretrained(
    model_path,
    trust_remote_code=True
)

model = AutoModel.from_pretrained(
    model_path,
    trust_remote_code=True,
    low_cpu_mem_usage=True
).float()

model = model.eval()

response, history = model.chat(
    tokenizer,
    "明明明明明白白白喜欢他，可她就是不说。 这句话里，明明和白白谁喜欢谁？",
    history=[]
)

print("\n模型回答：")
print(response)