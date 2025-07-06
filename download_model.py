from transformers import AutoModel , AutoTokenizer

# Download the model and tokenizer
model_name = "bert-base-uncased" #BERT Model
model = AutoModel.from_pretrained(model_name)
tokenizer = AutoTokenizer.from_pretrained(model_name)

# Use the model and tokenizer
inputs = tokenizer("Hello,hugging Face!",return_tensors="pt")
outputs = model(**inputs)
print(outputs.last_hidden_state.shape) #Eaxmple output Shape 