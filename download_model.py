from pathlib import Path
from transformers import AutoTokenizer, AutoModelForCausalLM

model_name = "Qwen/Qwen2.5-0.5B-Instruct"

# Store model downloads in your home folder, outside Google Drive.
cache = Path.home() / "jspace-data" / "models"
cache.mkdir(parents=True, exist_ok=True)

print("Downloading tokenizer...")
tokenizer = AutoTokenizer.from_pretrained(
    model_name,
    cache_dir=str(cache),
)

print("Downloading and loading model...")
model = AutoModelForCausalLM.from_pretrained(
    model_name,
    cache_dir=str(cache),
)

print("Model loaded successfully.")
print("Download location:", cache)
print("Transformer layers:", model.config.num_hidden_layers)
print("Residual dimension:", model.config.hidden_size)


import torch

# Move the model to your Mac's GPU.
model = model.to("mps")
model.eval()

messages = [
    {"role": "user", "content": "Name one sport. Answer in one word."}
]

text = tokenizer.apply_chat_template(
    messages,
    tokenize=False,
    add_generation_prompt=True,
)

inputs = tokenizer(
    text,
    return_tensors="pt",
    add_special_tokens=False,
).to("mps")

with torch.inference_mode():
    output = model.generate(
        **inputs,
        max_new_tokens=12,
        do_sample=False,
        pad_token_id=tokenizer.eos_token_id,
    )

# Display only the newly generated answer.
prompt_length = inputs["input_ids"].shape[1]
answer = output[0, prompt_length:]
print("Answer:", tokenizer.decode(answer, skip_special_tokens=True))