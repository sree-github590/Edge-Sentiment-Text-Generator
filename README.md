# Edge-Optimized Sentiment-Based Text Generator
A lightweight Generative AI model fine-tuned with LoRA on DistilGPT-2 to generate positive sentiment responses for edge devices.

## Setup
1. Clone: `git clone https://github.com/sree-github590/Edge-Sentiment-Text-Generator.git`
2. Install: `pip install -r requirements.txt`
3. Train: `python train.py`
4. Generate: `python generate.py`

## Features
- Fine-tuned DistilGPT-2 with LoRA for edge efficiency (~0.03% trainable parameters).
- Custom deep learning sentiment classifier (PyTorch) with 90% accuracy from 60% prior early training.
- Inference time: 0.696 seconds on CPU.

## Sample Output
Prompt: "I’m upset about my order."
Response: "I’m sorry to hear that—let me assist you today!"
Sentiment: Positive

## Technologies
- Python, PyTorch, Hugging Face, LoRA, NLP, Git
