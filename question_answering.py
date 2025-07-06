#pip install transformers torch
from transformers import AutoModelForQuestionAnswering , AutoTokenizer
model_name = "bert-large-uncased-whole-word-masking-finetuned-squad"
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModelForQuestionAnswering.from_pretrained(model_name)
context = """
Dr. A.P.J. Abdul Kalam (1931–2015), known as the Missile Man of India, was a renowned Indian aerospace scientist and the 11th President of India (2002–2007). Born in Rameswaram, Tamil Nadu, to a humble family, he rose through hard work and dedication. He played a key role in India’s civilian space program and military missile development, particularly through ISRO and DRDO.

Dr. Kalam was instrumental in India's 1998 nuclear tests and became a symbol of scientific innovation and national pride. Despite his scientific stature, he remained humble, deeply spiritual, and dedicated to youth and education.

As President, he was widely loved and respected, often called the "People’s President" for his connection with citizens, especially students. After his presidency, he continued to inspire millions through lectures, books, and mentorship.

He passed away on July 27, 2015, while delivering a lecture at IIM Shillong, leaving behind a legacy of vision, simplicity, and service.
"""

question = "Which Indian space and defense organizations did he work with?"
inputs = tokenizer(question,context,return_tensors="pt")
outputs = model(**inputs)


# Extract the start and End scores 
start_scores = outputs.start_logits
end_scores = outputs.end_logits

#Get the most likely start and end positions 

start_index = start_scores.argmax()
end_index = end_scores.argmax()

#  Convert token IDs back to words 

answer_tokens = inputs["input_ids"][0][start_index: end_index + 1]
answer = tokenizer.decode(answer_tokens, skip_special_tokens=True)

print(f"Answer: {answer}")
