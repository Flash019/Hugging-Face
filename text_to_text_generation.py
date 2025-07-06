#pip install transformers torch 
from transformers import T5ForConditionalGeneration, T5Tokenizer
#Load the pre-trained T5 Model and tokenizer
model_name= "t5-large" # You can use "t5-base", "t5-small"
tokenizer = T5Tokenizer.from_pretrained(model_name)
model = T5ForConditionalGeneration.from_pretrained(model_name)
input_text = """ Translate English to German: Bengali people love Rasgulla.
 It is a soft, round sweet made from milk and sugar. For them, Rasgulla is not just food—it is a part of their culture and happiness"""
 input_ids = tokenizer(input_text, return_tensors="pt").input_ids
 
#Translation Generation

outputs = model.generate(input_ids, max_length= 50)

#Decode the output  tokens to text

translated_text = tokenizer.decode(outputs[0], skip_special_tokens=True)

print("Translated Text:", translated_text)
