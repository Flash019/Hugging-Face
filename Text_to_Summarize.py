!pip install transformers torch
from transformers import AutoModelForSeq2SeqLM , AutoTokenizer
# Load a pre-trained model and tokenizer

model_name = "facebook/bart-large-cnn"  # Example model for Summarization 
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModelForSeq2SeqLM.from_pretrained(model_name)
# Input Text to Summarize 

text = """
Artificial Intelligence (AI) is rapidly transforming the healthcare industry by enhancing diagnostics, improving treatment plans, and streamlining administrative tasks. One of the most impactful applications is in medical imaging, where AI algorithms can detect abnormalities such as tumors, fractures, or infections with high accuracy, often surpassing human radiologists. These systems reduce diagnostic errors and speed up patient care.

AI is also revolutionizing personalized medicine. By analyzing large datasets, including genetic information and patient history, AI helps doctors tailor treatments to individual patients, increasing effectiveness and reducing side effects. In drug discovery, machine learning models predict how different compounds interact with the body, dramatically cutting down the time and cost needed to develop new medications.

Furthermore, AI chatbots and virtual health assistants support mental health care and patient engagement by providing 24/7 assistance and monitoring. Administrative burdens are also reduced, as AI automates scheduling, billing, and medical record management.

Despite these advancements, challenges such as data privacy, algorithm bias, and the need for regulatory frameworks must be addressed. With responsible development and deployment, AI has the potential to make healthcare more accurate, accessible, and efficient for everyone.

"""
# Tokenize the Input Text
inputs = tokenizer.encode("summarize: "+ text , return_tensors="pt", max_length=512, truncation=True)
inputs: Tokenized input text to summarize.

#max_length=50: Maximum number of tokens in the summary.

#min_length=25: Minimum number of tokens in the summary.

#length_penalty=2.0: Penalizes longer sequences to favor shorter summaries.

#num_beams=4: Uses beam search with 4 beams for better quality output.

#early_stopping=True: Stops when all beams finish generating an end token.
print(summary)
