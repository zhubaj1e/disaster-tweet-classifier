"""
Main Training Script for Disaster Tweet Classifier

This script trains a DistilBERT-based classifier on the Kaggle 
NLP with Disaster Tweets dataset.

Usage:
    python main.py

The trained model will be saved to the checkpoints/ directory.
"""

import os
import sys
import torch
import torch.nn as nn
import pandas as pd
import numpy as np
from collections import Counter
from tqdm import tqdm

from torch.optim import AdamW
from torch.utils.data import DataLoader
from transformers import DistilBertTokenizer, get_linear_schedule_with_warmup
from sklearn.model_selection import train_test_split
from sklearn.metrics import f1_score, classification_report

# Add src to path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from model import DisasterTweetClassifier
from utils import (
    preprocess_tweet, 
    DisasterTweetDataset,
    create_weighted_sampler,
    compute_class_weights,
    evaluate_model,
    set_seed,
    get_device
)
from config import (
    MODEL_NAME, MAX_LENGTH, BATCH_SIZE, LEARNING_RATE, EPOCHS,
    WARMUP_STEPS, DROPOUT, WEIGHT_DECAY, MAX_GRAD_NORM,
    TRAIN_DATA_PATH, VALIDATION_SPLIT, CHECKPOINT_DIR, 
    MODEL_FILENAME, SEED
)


def main():
    """Main training function."""
    
    # Set random seeds
    set_seed(SEED)
    
    # Get device
    device = get_device()
    
    # Load data
    print("\n" + "="*60)
    print("LOADING DATA")
    print("="*60)
    
    # Handle path relative to project root
    project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    train_path = os.path.join(project_root, TRAIN_DATA_PATH)
    
    if not os.path.exists(train_path):
        # Try relative to src folder
        train_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", TRAIN_DATA_PATH)
    
    train_df = pd.read_csv(train_path)
    print(f"Training samples: {len(train_df)}")
    
    # Preprocess
    print("\nPreprocessing tweets...")
    train_df['text_clean'] = train_df['text'].apply(preprocess_tweet)
    
    # Train/Validation Split
    train_texts, val_texts, train_labels, val_labels = train_test_split(
        train_df['text_clean'].tolist(),
        train_df['target'].tolist(),
        test_size=VALIDATION_SPLIT,
        random_state=SEED,
        stratify=train_df['target']
    )
    
    print(f"Training samples: {len(train_texts)}")
    print(f"Validation samples: {len(val_texts)}")
    print(f"Training class distribution: {Counter(train_labels)}")
    
    # Load tokenizer
    print("\nLoading tokenizer...")
    tokenizer = DistilBertTokenizer.from_pretrained(MODEL_NAME)
    
    # Create datasets
    train_dataset = DisasterTweetDataset(train_texts, train_labels, tokenizer, MAX_LENGTH)
    val_dataset = DisasterTweetDataset(val_texts, val_labels, tokenizer, MAX_LENGTH)
    
    # Create data loaders with weighted sampling
    sampler = create_weighted_sampler(train_labels)
    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, sampler=sampler)
    val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=False)
    
    print(f"Training batches: {len(train_loader)}")
    print(f"Validation batches: {len(val_loader)}")
    
    # Initialize model
    print("\n" + "="*60)
    print("INITIALIZING MODEL")
    print("="*60)
    
    model = DisasterTweetClassifier(num_classes=2, dropout=DROPOUT).to(device)
    print(f"Model parameters: {sum(p.numel() for p in model.parameters()):,}")
    
    # Training setup
    class_weights = compute_class_weights(train_labels, device)
    loss_fn = nn.CrossEntropyLoss(weight=class_weights)
    optimizer = AdamW(model.parameters(), lr=LEARNING_RATE, weight_decay=WEIGHT_DECAY)
    
    total_steps = len(train_loader) * EPOCHS
    scheduler = get_linear_schedule_with_warmup(
        optimizer, 
        num_warmup_steps=WARMUP_STEPS, 
        num_training_steps=total_steps
    )
    
    print(f"\nTraining Configuration:")
    print(f"  Learning Rate: {LEARNING_RATE}")
    print(f"  Batch Size: {BATCH_SIZE}")
    print(f"  Epochs: {EPOCHS}")
    print(f"  Dropout: {DROPOUT}")
    print(f"  Total steps: {total_steps}")
    
    # Training loop
    print("\n" + "="*60)
    print("TRAINING")
    print("="*60)
    
    best_val_f1 = 0
    best_model_state = None
    history = {'train_loss': [], 'val_loss': [], 'train_f1': [], 'val_f1': []}
    
    for epoch in range(EPOCHS):
        model.train()
        running_loss = 0.0
        train_preds, train_labels_epoch = [], []
        
        progress_bar = tqdm(train_loader, desc=f"Epoch {epoch+1}/{EPOCHS}")
        for batch in progress_bar:
            input_ids = batch['input_ids'].to(device)
            attention_mask = batch['attention_mask'].to(device)
            labels = batch['labels'].to(device)
            
            optimizer.zero_grad()
            outputs = model(input_ids, attention_mask)
            loss = loss_fn(outputs, labels)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=MAX_GRAD_NORM)
            optimizer.step()
            scheduler.step()
            
            running_loss += loss.item()
            preds = torch.argmax(outputs, dim=1).cpu().numpy()
            train_preds.extend(preds)
            train_labels_epoch.extend(labels.cpu().numpy())
            progress_bar.set_postfix({'loss': f'{loss.item():.4f}'})
        
        # Calculate metrics
        train_loss = running_loss / len(train_loader)
        train_f1 = f1_score(train_labels_epoch, train_preds, average='binary')
        val_results = evaluate_model(model, val_loader, loss_fn, device)
        
        # Store history
        history['train_loss'].append(train_loss)
        history['val_loss'].append(val_results['loss'])
        history['train_f1'].append(train_f1)
        history['val_f1'].append(val_results['f1'])
        
        print(f"\nEpoch {epoch+1}/{EPOCHS}")
        print(f"  Train Loss: {train_loss:.4f} | Train F1: {train_f1:.4f}")
        print(f"  Val Loss: {val_results['loss']:.4f} | Val F1: {val_results['f1']:.4f}")
        
        # Save best model
        if val_results['f1'] > best_val_f1:
            best_val_f1 = val_results['f1']
            best_model_state = model.state_dict().copy()
            print(f"  *** New best model (Val F1: {best_val_f1:.4f}) ***")
    
    # Save model
    print("\n" + "="*60)
    print("SAVING MODEL")
    print("="*60)
    
    checkpoint_path = os.path.join(project_root, CHECKPOINT_DIR)
    os.makedirs(checkpoint_path, exist_ok=True)
    model_path = os.path.join(checkpoint_path, MODEL_FILENAME)
    
    torch.save(best_model_state, model_path)
    print(f"Model saved to: {model_path}")
    
    # Final evaluation
    print("\n" + "="*60)
    print("FINAL EVALUATION")
    print("="*60)
    
    model.load_state_dict(best_model_state)
    final_results = evaluate_model(model, val_loader, loss_fn, device)
    
    print(f"Best Validation F1: {final_results['f1']:.4f}")
    print(f"Validation Accuracy: {final_results['accuracy']:.4f}")
    print(f"Validation Precision: {final_results['precision']:.4f}")
    print(f"Validation Recall: {final_results['recall']:.4f}")
    
    print("\nClassification Report:")
    print(classification_report(
        final_results['labels'], 
        final_results['predictions'],
        target_names=['Not Disaster', 'Disaster']
    ))
    
    print("\nTraining complete!")
    
    return model, history


if __name__ == "__main__":
    main()
