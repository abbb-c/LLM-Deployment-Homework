# 大语言模型部署体验实验

## 1 项目简介

本实验基于 ModelScope 平台完成多个大语言模型的部署与推理测试，并对不同模型在中文语义理解任务中的表现进行横向对比分析。

实验中使用的模型包括：

- ChatGLM3-6B
- Qwen-7B-Chat
- Baichuan2-7B-Chat

实验环境为 ModelScope 提供的 CPU 云服务器环境，通过 Python 与 transformers 框架完成模型加载与推理。

---

## 2 实验环境

### 2.1 平台环境

- ModelScope Notebook
- Linux CPU 云服务器

### 2.2 Python 环境

安装依赖：

```bash
pip install transformers==4.30.2
pip install torch sentencepiece accelerate
```

---

## 3 模型下载

在 `/mnt/data` 目录下下载模型：

```bash
git clone https://www.modelscope.cn/ZhipuAI/chatglm3-6b.git

git clone https://www.modelscope.cn/qwen/Qwen-7B-Chat.git

git clone https://www.modelscope.cn/baichuan-inc/Baichuan2-7B-Chat.git
```

---

## 4 模型推理

使用 transformers 加载模型并完成推理测试。

示例代码：

```python
from transformers import AutoTokenizer, AutoModel
import torch

model_path = "/mnt/data/chatglm3-6b"

tokenizer = AutoTokenizer.from_pretrained(
    model_path,
    trust_remote_code=True
)

model = AutoModel.from_pretrained(
    model_path,
    trust_remote_code=True
).float()

model.eval()

response, history = model.chat(
    tokenizer,
    "请说出以下两句话区别在哪里？1、冬天：能穿多少穿多少 2、夏天：能穿多少穿多少",
    history=[]
)

print(response)
```

---

## 5 测试问题

实验中使用了多个中文语义理解问题进行测试，包括：

- 请说出以下两句话区别在哪里？
  1、冬天：能穿多少穿多少
  2、夏天：能穿多少穿多少

- 明明明明明白白白喜欢他，可她就是不说。这句话里，明明和白白谁喜欢谁？

---

## 6 横向对比分析

### ChatGLM3-6B

模型输出较自然，在基础中文语义理解任务中表现较稳定，但在复杂歧义句推理任务中容易直接进行主观推断。

### Qwen-7B-Chat

模型回答较详细，对于歧义问题表现出较强的谨慎性，但模型资源占用较高，在 CPU 环境下运行时容易出现内存压力。

### Baichuan2-7B-Chat

模型整体运行较稳定，能够识别部分复杂语义问题中的歧义现象，但回答相对简洁。

---

## 7 实验总结

本实验完成了多个大语言模型在 CPU 环境下的部署与推理测试，并对不同模型在中文语义理解任务中的表现进行了横向对比。

实验过程中学习了模型下载、环境配置、模型推理以及内存优化等内容，对大语言模型的实际部署流程有了更加深入的理解。

具体可见实验报告。
