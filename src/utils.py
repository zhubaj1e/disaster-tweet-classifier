"""
Utility Functions for Disaster Tweet Classification

This module contains helper functions for data preprocessing, 
dataset creation, and model evaluation.
"""

import re
import torch
import numpy as np
import pandas as pd
from typing import List, Dict, Tuple, Optional
from collections import Counter
from torch.utils.data import Dataset, WeightedRandomSampler
from sklearn.metrics import f1_score, precision_score, recall_score, confusion_matrix


def preprocess_tweet(text: str) -> str:
    """
    Preprocess tweet text for model input.
    
    Preprocessing steps:
    1. Remove URLs (no semantic value)
    2. Remove @mentions (usernames not relevant)
    3. Preserve hashtag words (contain useful keywords)
    4. Clean special characters
    
    Args:
        text: Raw tweet text
        
    Returns:
        Cleaned tweet text
    """
    if pd.isna(text):
        return ""
    
    text = str(text)
    
    # Remove URLs
    text = re.sub(r'http\S+|www\S+|https\S+', '', text, flags=re.MULTILINE)
    
    # Remove mentions
    text = re.sub(r'@\w+', '', text)
    
    # Convert hashtags to words (remove # but keep the word)
    text = re.sub(r'#(\w+)', r'\1', text)
    
    # Remove special characters (keep alphanumeric and basic punctuation)
    text = re.sub(r'[^\w\s.,!?]', ' ', text)
    
    # Clean up whitespace
    text = ' '.join(text.split())
    
    return text.strip()


class DisasterTweetDataset(Dataset):
    """
    PyTorch Dataset for Disaster Tweets.
    
    Args:
        texts: List of tweet texts
        labels: List of labels (0 = Not Disaster, 1 = Disaster)
        tokenizer: HuggingFace tokenizer instance
        max_length: Maximum sequence length for tokenization
    """
    
    def __init__(self, texts: List[str], labels: List[int], tokenizer, max_length: int = 128):
        self.texts = texts
        self.labels = labels
        self.tokenizer = tokenizer
        self.max_length = max_length
    
    def __len__(self) -> int:
        return len(self.texts)
    
    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        encoding = self.tokenizer(
            str(self.texts[idx]),
            padding='max_length',
            truncation=True,
            max_length=self.max_length,
            return_tensors='pt'
        )
        
        return {
            'input_ids': encoding['input_ids'].squeeze(0),
            'attention_mask': encoding['attention_mask'].squeeze(0),
            'labels': torch.tensor(self.labels[idx], dtype=torch.long)
        }


def create_weighted_sampler(labels: List[int]) -> WeightedRandomSampler:
    """
    Create a weighted random sampler for handling class imbalance.
    
    Args:
        labels: List of class labels
        
    Returns:
        WeightedRandomSampler instance
    """
    class_counts = Counter(labels)
    sample_weights = [1.0 / class_counts[label] for label in labels]
    return WeightedRandomSampler(sample_weights, len(sample_weights), replacement=True)


def compute_class_weights(labels: List[int], device: torch.device) -> torch.Tensor:
    """
    Compute class weights for balanced loss function.
    
    Args:
        labels: List of class labels
        device: Device to create tensor on
        
    Returns:
        Tensor of class weights
    """
    class_counts = Counter(labels)
    total_samples = len(labels)
    
    weight_0 = total_samples / (2 * class_counts[0])
    weight_1 = total_samples / (2 * class_counts[1])
    
    return torch.tensor([weight_0, weight_1], dtype=torch.float).to(device)


def evaluate_model(model, data_loader, loss_fn, device: torch.device) -> Dict:
    """
    Evaluate model on a dataset.
    
    Args:
        model: PyTorch model
        data_loader: DataLoader for evaluation data
        loss_fn: Loss function
        device: Device for computation
        
    Returns:
        Dictionary containing evaluation metrics
    """
    model.eval()
    total_loss = 0
    all_preds, all_labels = [], []
    
    with torch.no_grad():
        for batch in data_loader:
            input_ids = batch['input_ids'].to(device)
            attention_mask = batch['attention_mask'].to(device)
            labels = batch['labels'].to(device)
            
            outputs = model(input_ids, attention_mask)
            loss = loss_fn(outputs, labels)
            total_loss += loss.item()
            
            preds = torch.argmax(outputs, dim=1).cpu().numpy()
            all_preds.extend(preds)
            all_labels.extend(labels.cpu().numpy())
    
    all_preds = np.array(all_preds)
    all_labels = np.array(all_labels)
    
    return {
        'loss': total_loss / len(data_loader),
        'accuracy': (all_preds == all_labels).mean(),
        'f1': f1_score(all_labels, all_preds, average='binary'),
        'precision': precision_score(all_labels, all_preds, average='binary'),
        'recall': recall_score(all_labels, all_preds, average='binary'),
        'predictions': all_preds.tolist(),
        'labels': all_labels.tolist()
    }


def predict_single_tweet(model, tokenizer, tweet: str, device: torch.device, 
                         max_length: int = 128) -> Tuple[str, float]:
    """
    Make prediction for a single tweet.
    
    Args:
        model: Trained model
        tokenizer: Tokenizer instance
        tweet: Raw tweet text
        device: Device for computation
        max_length: Maximum sequence length
        
    Returns:
        Tuple of (prediction label, confidence score)
    """
    # Preprocess
    clean_tweet = preprocess_tweet(tweet)
    
    # Tokenize
    encoding = tokenizer(
        clean_tweet,
        padding='max_length',
        truncation=True,
        max_length=max_length,
        return_tensors='pt'
    )
    
    # Predict
    model.eval()
    with torch.no_grad():
        input_ids = encoding['input_ids'].to(device)
        attention_mask = encoding['attention_mask'].to(device)
        outputs = model(input_ids, attention_mask)
        probs = torch.softmax(outputs, dim=1)
        pred = torch.argmax(outputs, dim=1).item()
        confidence = probs[0][pred].item()
    
    label = "DISASTER" if pred == 1 else "NOT DISASTER"
    return label, confidence


def set_seed(seed: int = 42):
    """
    Set random seeds for reproducibility.
    
    Args:
        seed: Random seed value
    """
    import random
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def get_device() -> torch.device:
    """
    Get the best available device for computation.
    
    Returns:
        torch.device object
    """
    if torch.cuda.is_available():
        device = torch.device("cuda")
        print(f"Using CUDA: {torch.cuda.get_device_name(0)}")
    elif torch.backends.mps.is_available():
        device = torch.device("mps")
        print("Using MPS (Apple Silicon)")
    else:
        device = torch.device("cpu")
        print("Using CPU")
    
    return device
