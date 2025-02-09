import os
import json
from argparse import ArgumentParser
import random
import logging
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from tqdm import tqdm
    
import torch
from torch.utils.data import DataLoader, Dataset, random_split
from transformers import AutoTokenizer, get_linear_schedule_with_warmup, AdamW

from DeepSeek.model import Transformer, ModelArgs

class JsonDataset(Dataset):
    def __init__(self, json_folder, tokenizer, max_length=512):
        self.json_folder = json_folder
        self.tokenizer = tokenizer
        self.max_length = max_length
        self.data = self.load_json_files()
        self.randomize_labels()

    def load_json_files(self):
        data = []
        for filename in os.listdir(self.json_folder):
            if filename.endswith(".json"):
                filepath = os.path.join(self.json_folder, filename)
                with open(filepath, 'r') as f:
                    json_data = json.load(f)
                    content = json_data['choices'][0]['message']['content']
                    citations = json_data['citations']
                    data.append((content, citations, True))  # Initially label all as True
        return data

    def randomize_labels(self):
        num_false = int(0.2 * len(self.data))
        false_indices = random.sample(range(len(self.data)), num_false)
        for idx in false_indices:
            self.data[idx] = (self.data[idx][0], self.data[idx][1], False)

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        content, citations, label = self.data[idx]
        inputs = self.tokenizer(content, padding='max_length', truncation=True, max_length=self.max_length, return_tensors="pt")
        labels = self.tokenizer(citations, padding='max_length', truncation=True, max_length=self.max_length, return_tensors="pt")
        return {
            'input_ids': inputs['input_ids'].squeeze(),
            'attention_mask': inputs['attention_mask'].squeeze(),
            'labels': labels['input_ids'].squeeze(),
            'label': torch.tensor(label, dtype=torch.float)
        }

def train(model, dataloader, optimizer, scheduler, device, grad_clip):
    model.train()
    total_loss = 0
    all_preds = []
    all_labels = []
    for batch in tqdm(dataloader, desc="Training"):
        inputs = batch['input_ids'].to(device)
        labels = batch['labels'].to(device)
        label = batch['label'].to(device)
        optimizer.zero_grad()
        outputs = model(inputs)
        loss = torch.nn.functional.cross_entropy(outputs.view(-1, outputs.size(-1)), labels.view(-1))
        loss.backward()
        if grad_clip > 0:
            torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip)
        optimizer.step()
        scheduler.step()
        total_loss += loss.item()
        preds = torch.argmax(outputs, dim=-1).view(-1).cpu().numpy()
        all_preds.extend(preds)
        all_labels.extend(labels.view(-1).cpu().numpy())
    accuracy = accuracy_score(all_labels, all_preds)
    precision = precision_score(all_labels, all_preds, average='weighted')
    recall = recall_score(all_labels, all_preds, average='weighted')
    f1 = f1_score(all_labels, all_preds, average='weighted')
    return total_loss / len(dataloader), accuracy, precision, recall, f1

def validate(model, dataloader, device):
    model.eval()
    total_loss = 0
    all_preds = []
    all_labels = []
    with torch.no_grad():
        for batch in tqdm(dataloader, desc="Validation"):
            inputs = batch['input_ids'].to(device)
            labels = batch['labels'].to(device)
            outputs = model(inputs)
            loss = torch.nn.functional.cross_entropy(outputs.view(-1, outputs.size(-1)), labels.view(-1))
            total_loss += loss.item()
            preds = torch.argmax(outputs, dim=-1).view(-1).cpu().numpy()
            all_preds.extend(preds)
            all_labels.extend(labels.view(-1).cpu().numpy())
    accuracy = accuracy_score(all_labels, all_preds)
    precision = precision_score(all_labels, all_preds, average='weighted')
    recall = recall_score(all_labels, all_preds, average='weighted')
    f1 = f1_score(all_labels, all_preds, average='weighted')
    return total_loss / len(dataloader), accuracy, precision, recall, f1

def main(args):
    logging.basicConfig(level=logging.INFO)
    device = torch.device(args.device)
    tokenizer = AutoTokenizer.from_pretrained(args.model_name_or_path)
    dataset = JsonDataset(args.json_folder, tokenizer)
    val_size = int(len(dataset) * args.validation_split)
    train_size = len(dataset) - val_size
    train_dataset, val_dataset = random_split(dataset, [train_size, val_size])
    train_dataloader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True)
    val_dataloader = DataLoader(val_dataset, batch_size=args.batch_size, shuffle=False)

    with open(args.config) as f:
        model_args = ModelArgs(**json.load(f))
    model = Transformer(model_args).to(device)

    optimizer = AdamW(model.parameters(), lr=args.learning_rate, weight_decay=args.weight_decay)
    total_steps = len(train_dataloader) * args.epochs
    scheduler = get_linear_schedule_with_warmup(optimizer, num_warmup_steps=args.warmup_steps, num_training_steps=total_steps)

    best_f1 = 0
    metrics = {"train_loss": [], "val_loss": [], "train_acc": [], "val_acc": [], "train_prec": [], "val_prec": [], "train_rec": [], "val_rec": [], "train_f1": [], "val_f1": []}
    for epoch in range(args.epochs):
        train_loss, train_acc, train_prec, train_rec, train_f1 = train(model, train_dataloader, optimizer, scheduler, device, args.grad_clip)
        val_loss, val_acc, val_prec, val_rec, val_f1 = validate(model, val_dataloader, device)
        logging.info(f"Epoch {epoch + 1}/{args.epochs}, Train Loss: {train_loss}, Val Loss: {val_loss}")
        logging.info(f"Train Metrics - Acc: {train_acc}, Prec: {train_prec}, Rec: {train_rec}, F1: {train_f1}")
        logging.info(f"Val Metrics - Acc: {val_acc}, Prec: {val_prec}, Rec: {val_rec}, F1: {val_f1}")

        metrics["train_loss"].append(train_loss)
        metrics["val_loss"].append(val_loss)
        metrics["train_acc"].append(train_acc)
        metrics["val_acc"].append(val_acc)
        metrics["train_prec"].append(train_prec)
        metrics["val_prec"].append(val_prec)
        metrics["train_rec"].append(train_rec)
        metrics["val_rec"].append(val_rec)
        metrics["train_f1"].append(train_f1)
        metrics["val_f1"].append(val_f1)

        if val_f1 > best_f1:
            best_f1 = val_f1
            model.save_pretrained(args.output_dir)
            tokenizer.save_pretrained(args.output_dir)
            logging.info(f"Model saved to {args.output_dir}")

        # Early stopping
        if epoch > 0 and val_loss >= best_f1:
            logging.info("Early stopping")
            break

    with open(os.path.join(args.output_dir, "metrics.json"), 'w') as f:
        json.dump(metrics, f)

if __name__ == "__main__":
    parser = ArgumentParser()
    parser.add_argument("--model_name_or_path", type=str, required=True, help="Path to pre-trained model or model identifier from huggingface.co/models")
    parser.add_argument("--dataset_name", type=str, required=True, help="Name of the dataset to use")
    parser.add_argument("--config", type=str, required=True, help="Path to the model configuration file")
    parser.add_argument("--output_dir", type=str, required=True, help="Directory to save the trained model")
    parser.add_argument("--batch_size", type=int, default=8, help="Batch size for training")
    parser.add_argument("--learning_rate", type=float, default=5e-5, help="Learning rate")
    parser.add_argument("--epochs", type=int, default=3, help="Number of training epochs")
    parser.add_argument("--warmup_steps", type=int, default=0, help="Number of warmup steps")
    parser.add_argument("--json_folder", type=str, required=True, help="Path to the folder containing JSON files")
    parser.add_argument("--validation_split", type=float, default=0.1, help="Fraction of data to use for validation")
    parser.add_argument("--log_interval", type=int, default=10, help="Interval for logging training progress")
    parser.add_argument("--device", type=str, default="cpu", help="Device to use for training (cpu, cuda, mps)")
    parser.add_argument("--weight_decay", type=float, default=0.01, help="Weight decay for optimizer")
    parser.add_argument("--grad_clip", type=float, default=1.0, help="Gradient clipping value")
    args = parser.parse_args()
    main(args)