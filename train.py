from datasets import load_dataset
import torch
import torch.nn as nn
from sklearn.feature_extraction.text import TfidfVectorizer
import numpy as np
import joblib  # Add this import

# Load custom text dataset
dataset = load_dataset('text', data_files='responses.txt')
dataset = dataset['train'].train_test_split(test_size=0.1)  # 90% train, 10% test
print(f"Dataset loaded: {len(dataset['train'])} train, {len(dataset['test'])} test samples")

# Define a simple deep learning sentiment classifier
class SentimentClassifier(nn.Module):
    def __init__(self, input_dim):
        super(SentimentClassifier, self).__init__()
        self.fc1 = nn.Linear(input_dim, 128)
        self.dropout = nn.Dropout(0.3)
        self.fc2 = nn.Linear(128, 64)
        self.fc3 = nn.Linear(64, 2)  # 2 classes: positive (1), negative (0)

    def forward(self, x):
        x = torch.relu(self.fc1(x))
        x = self.dropout(x)
        x = torch.relu(self.fc2(x))
        x = self.fc3(x)
        return x

# Prepare synthetic training data (positive vs. negative)
positive_texts = dataset['train']['text']
negative_texts = ["I hate this service!", "This is terrible!", "I’m so angry!"]
all_texts = positive_texts + negative_texts
labels = [1] * len(positive_texts) + [0] * len(negative_texts)

# Vectorize text
vectorizer = TfidfVectorizer(max_features=500)
X = vectorizer.fit_transform(all_texts).toarray()
y = torch.LongTensor(labels)

# Get the actual number of features
input_dim = X.shape[1]  # Dynamically set based on vectorizer output
print(f"Input dimension: {input_dim} features")

# Save the vectorizer for consistency
joblib.dump(vectorizer, 'tfidf_vectorizer.pkl')

# Train classifier
classifier = SentimentClassifier(input_dim=input_dim)  # Use dynamic input_dim
optimizer = torch.optim.Adam(classifier.parameters(), lr=0.001)
criterion = nn.CrossEntropyLoss()

X_tensor = torch.FloatTensor(X)
for epoch in range(50):  # Small epochs for quick training
    optimizer.zero_grad()
    outputs = classifier(X_tensor)
    loss = criterion(outputs, y)
    loss.backward()
    optimizer.step()
    if epoch % 10 == 0:
        print(f"Classifier Epoch {epoch}, Loss: {loss.item():.4f}")

# Save classifier
torch.save(classifier.state_dict(), 'sentiment_classifier.pt')

# --- Previous Step 7 code appended ---
from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import LoraConfig, get_peft_model
from torch.utils.data import DataLoader
import torch.optim as optim

# Load DistilGPT-2 (lightweight for edge)
model_name = "distilgpt2"
tokenizer = AutoTokenizer.from_pretrained(model_name)
tokenizer.pad_token = tokenizer.eos_token  # Set padding token
model = AutoModelForCausalLM.from_pretrained(model_name)

# Apply LoRA for efficient fine-tuning
lora_config = LoraConfig(
    r=8,  # Low rank for efficiency
    lora_alpha=32,
    target_modules=["h.0.attn.c_attn"],  # Specific to DistilGPT-2 layers
    lora_dropout=0.1,
    task_type="CAUSAL_LM"
)
model = get_peft_model(model, lora_config)
model.print_trainable_parameters()  # Check trainable parameters (should be ~1-2% of total)

def tokenize_function(examples):
    return tokenizer(examples["text"], padding="max_length", truncation=True, max_length=64)

tokenized_dataset = dataset.map(tokenize_function, batched=True)
tokenized_dataset.set_format('torch', columns=['input_ids', 'attention_mask'])

# DataLoader
train_loader = DataLoader(tokenized_dataset['train'], batch_size=4, shuffle=True)

# Training setup
optimizer = optim.AdamW(model.parameters(), lr=5e-5)
model.train()

# Training loop
for epoch in range(3):  # 3 epochs for small dataset
    total_loss = 0
    for batch in train_loader:
        optimizer.zero_grad()
        input_ids = batch['input_ids']
        attention_mask = batch['attention_mask']
        outputs = model(input_ids, attention_mask=attention_mask, labels=input_ids)
        loss = outputs.loss
        loss.backward()
        optimizer.step()
        total_loss += loss.item()
    avg_loss = total_loss / len(train_loader)
    print(f"Epoch {epoch+1}, Avg Loss: {avg_loss:.4f}")

# Save fine-tuned model
model.save_pretrained("fine_tuned_distilgpt2")
tokenizer.save_pretrained("fine_tuned_distilgpt2")