
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM
# Load a translation model (English to Arabic)
model_name = "Helsinki-NLP/opus-mt-en-ar"
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModelForSeq2SeqLM.from_pretrained(model_name)

# Input English text
input_text = """Translate English to Arabic: Bengali people love Rasgulla.
It is a soft, round sweet made from milk and sugar. For them, Rasgulla is not just food—it is a part of their culture and happiness."""

# Tokenize input
input_ids = tokenizer(input_text, return_tensors="pt").input_ids

# Generate translation
outputs = model.generate(input_ids, max_length=50)

# Decode and print result
translated_text = tokenizer.decode(outputs[0], skip_special_tokens=True)
print("Translated Text:", translated_text)
