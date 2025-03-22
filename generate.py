import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
import torch.nn as nn
from sklearn.feature_extraction.text import TfidfVectorizer
import joblib

# Load fine-tuned model and tokenizer
model = AutoModelForCausalLM.from_pretrained("fine_tuned_distilgpt2")
tokenizer = AutoTokenizer.from_pretrained("fine_tuned_distilgpt2")

# Load sentiment classifier
class SentimentClassifier(nn.Module):
    def __init__(self, input_dim):
        super(SentimentClassifier, self).__init__()
        self.fc1 = nn.Linear(input_dim, 128)
        self.dropout = nn.Dropout(0.3)
        self.fc2 = nn.Linear(128, 64)
        self.fc3 = nn.Linear(64, 2)

    def forward(self, x):
        x = torch.relu(self.fc1(x))
        x = self.dropout(x)
        x = torch.relu(self.fc2(x))
        x = self.fc3(x)
        return x

# Load the saved vectorizer from training
vectorizer = joblib.load('tfidf_vectorizer.pkl')
input_dim = len(vectorizer.get_feature_names_out())  # Match training input_dim (163)
print(f"Input dimension in generate.py: {input_dim}")

# Load classifier
classifier = SentimentClassifier(input_dim=input_dim)
classifier.load_state_dict(torch.load('sentiment_classifier.pt', weights_only=True))
classifier.eval()

# Generate response
def generate_response(prompt):
    model.eval()
    inputs = tokenizer(prompt, return_tensors="pt", padding=True, truncation=True, max_length=64)
    outputs = model.generate(
        **inputs,
        max_new_tokens=30,
        do_sample=True,
        top_k=50,
        top_p=0.95,
        temperature=0.7
    )
    response = tokenizer.decode(outputs[0], skip_special_tokens=True)
    return response

# Validate sentiment
def validate_sentiment(text):
    vector = vectorizer.transform([text]).toarray()
    tensor = torch.FloatTensor(vector)
    with torch.no_grad():
        sentiment = classifier(tensor).argmax(dim=1).item()
    return "Positive" if sentiment == 1 else "Negative"

# Test
prompt = "I’m upset about my order."
response = generate_response(prompt)
sentiment = validate_sentiment(response)
print(f"Prompt: {prompt}")
print(f"Response: {response}")
print(f"Sentiment: {sentiment}")

import time

# Benchmark inference
start_time = time.time()
response = generate_response(prompt)
inference_time = time.time() - start_time
print(f"Inference Time: {inference_time:.4f} seconds")


