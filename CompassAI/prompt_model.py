import os
import json
import torch
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM

def load_model_and_tokenizer(model_dir):
    tokenizer = AutoTokenizer.from_pretrained(model_dir)
    model = AutoModelForSeq2SeqLM.from_pretrained(model_dir)
    return model, tokenizer

def generate_predictions(model, tokenizer, input_texts, device):
    model.to(device)
    model.eval()
    inputs = tokenizer(input_texts, return_tensors="pt", padding=True, truncation=True).to(device)
    with torch.no_grad():
        outputs = model.generate(**inputs)
    predictions = [tokenizer.decode(output, skip_special_tokens=True) for output in outputs]
    return predictions

if __name__ == "__main__":
    model_dir = "path/to/saved/model"  # Replace with the actual path to the saved model directory
    input_texts = ["Example input text 1", "Example input text 2"]  # Replace with actual input texts
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    model, tokenizer = load_model_and_tokenizer(model_dir)
    predictions = generate_predictions(model, tokenizer, input_texts, device)
    
    for i, prediction in enumerate(predictions):
        print(f"Input: {input_texts[i]}")
        print(f"Prediction: {prediction}")
        print()