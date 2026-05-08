# 🤖 TinyLlama Text-to-SQL Assistant

[![Model: TinyLlama 1.1B](https://img.shields.io/badge/Model-TinyLlama_1.1B-blue.svg)](https://huggingface.co/TinyLlama/TinyLlama-1.1B-Chat-v1.0)
[![Technique: LoRA](https://img.shields.io/badge/Fine--Tuning-LoRA-green.svg)](https://huggingface.co/docs/peft/index)
[![Dataset: SQL Create Context](https://img.shields.io/badge/Dataset-b--mc2%2Fsql--create--context-orange.svg)](https://huggingface.co/datasets/b-mc2/sql-create-context)

## 📌 Overview
This repository contains the code and fine-tuned adapter for a **Text-to-SQL** assistant. The model takes a database table schema and a natural language question, and generates the corresponding, precise SQL query. 

It was trained by fine-tuning the `TinyLlama-1.1B-Chat-v1.0` model using **LoRA (Low-Rank Adaptation)**, making it highly efficient, lightweight, and capable of running on consumer-grade GPUs like the T4.

## 🚀 Features
- **Efficient Fine-Tuning:** Trained using pure LoRA (no QLoRA/4-bit quantization) targeting attention projections.
- **Lightweight:** The resulting adapter is only ~20 MB in size.
- **High Precision:** Transforms a chatty base model into a strict, accurate SQL code generator.
- **Fast Inference:** Suitable for deployment on free CPU/GPU tiers (e.g., Hugging Face Spaces).

## 🧠 Training Details
- **Base Model:** `TinyLlama/TinyLlama-1.1B-Chat-v1.0`
- **Dataset:** A subset (3,000 examples) of `b-mc2/sql-create-context`
- **Hardware:** 1x NVIDIA Tesla T4 (16GB VRAM)
- **LoRA Config:** `r=16`, `alpha=32`, `dropout=0.05`
- **Target Modules:** `q_proj`, `k_proj`, `v_proj`, `o_proj`
- **Training Time:** ~5-8 minutes (1 Epoch)

## 💡 Example: Before vs. After
**Schema:** `CREATE TABLE employees (id INT, name TEXT, department TEXT, salary INT);`
**Question:** *List the names of employees in the Engineering department earning more than 100000.*

| Model State | Output |
| :--- | :--- |
| **Base Model (Before)** | *To answer this question, you can use the following SQL query: `SELECT name FROM employees...` This query will return a list of names...* (Chatty and ignores formatting instructions) |
| **Fine-Tuned (After)** | `SELECT name FROM employees WHERE department = "engineering" AND salary > 100000` (Strict, accurate, and direct) |

## 🛠️ Usage (Inference)

You can easily load this adapter on top of the base model using the `transformers` and `peft` libraries:

```python
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import PeftModel

BASE_MODEL = "TinyLlama/TinyLlama-1.1B-Chat-v1.0"
ADAPTER_DIR = "your-username/tinyllama-sql-lora" # Replace with your HF repo

# 1. Load Tokenizer & Base Model
tokenizer = AutoTokenizer.from_pretrained(BASE_MODEL)
base_model = AutoModelForCausalLM.from_pretrained(
    BASE_MODEL,
    device_map="auto",
    torch_dtype=torch.float16,
)

# 2. Attach LoRA Adapter
model = PeftModel.from_pretrained(base_model, ADAPTER_DIR)
model.eval()

# 3. Generate SQL
schema = "CREATE TABLE users (id INT, name TEXT, age INT);"
question = "Get the names of users older than 30."

prompt = f"<|system|>\nYou are a SQL assistant. Given a table schema and a question, reply with ONLY the SQL query, nothing else.</s>\n<|user|>\nSchema:\n{schema}\n\nQuestion: {question}</s>\n<|assistant|>\n"

inputs = tokenizer(prompt, return_tensors="pt").to(model.device)
with torch.no_grad():
    outputs = model.generate(**inputs, max_new_tokens=100, do_sample=False)

print(tokenizer.decode(outputs[0][inputs.input_ids.shape[1]:], skip_special_tokens=True).strip())
