from transformers import GPT2LMHeadModel, GPT2Tokenizer

# Load model & tokenizer
tokenizer = GPT2Tokenizer.from_pretrained("gpt2")
model = GPT2LMHeadModel.from_pretrained("gpt2")

prompt = "Artificial Intelligence is transforming the world because"

inputs = tokenizer(prompt, return_tensors="pt")

outputs = model.generate(
    inputs["input_ids"],
    max_length=80,
    num_return_sequences=1,
    no_repeat_ngram_size=2,
    temperature=0.8
)

generated_text = tokenizer.decode(outputs[0], skip_special_tokens=True)

print(generated_text)
