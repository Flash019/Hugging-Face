#pip install transformers
#pip install datasets
#pip install torch
from transformers import pipeline
spam_classifier = pipeline(
    "text-classification",
    model="philschmid/distilbert-base-multilingual-cased-sentiment",
    framework="pt")
texts = [
    "Congratulations! You've won a ₹10,000 gift card. Click here to claim it.",
    "Earn ₹50,000 per day from home without doing anything.",
    "Get instant loan approval without documents.",
    "Let’s meet at 5 PM near the coffee shop.",
    "The weather looks great today. Want to go for a walk?"
]

label_mapping = {'negative': 'SPAM', 'neutral': 'NOT SPAM', 'positive': 'NOT SPAM'}

results = spam_classifier(texts)

for result in results:
    label = label_mapping.get(result['label'], 'UNKNOWN')
    score = result['score']
    print(f"Label: {label}, Confidence: {score:.4f}")
