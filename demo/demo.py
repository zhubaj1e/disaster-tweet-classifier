"""
Disaster Tweet Classifier - Demo Script

This script demonstrates the core functionality of the Disaster Tweet Classifier.
Run from the project root directory:
    python demo/demo.py

Or from the demo directory:
    python demo.py
"""

import os
import sys

# Get project root directory
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.dirname(SCRIPT_DIR)

# Add src to path
sys.path.insert(0, os.path.join(PROJECT_ROOT, 'src'))

import torch
from transformers import DistilBertTokenizer
from model import load_model
from utils import predict_single_tweet, get_device
from config import MODEL_NAME, MAX_LENGTH, DROPOUT


def main():
    """Run the demo."""
    print("=" * 70)
    print("DISASTER TWEET CLASSIFIER - DEMO")
    print("=" * 70)
    
    # Get device
    device = get_device()
    
    # Load tokenizer
    print("\nLoading tokenizer...")
    tokenizer = DistilBertTokenizer.from_pretrained(MODEL_NAME)
    print(f"Tokenizer loaded: {MODEL_NAME}")
    
    # Load model
    print("\nLoading model...")
    checkpoint_path = os.path.join(PROJECT_ROOT, 'checkpoints', 'disaster_tweet_classifier.pt')
    
    if not os.path.exists(checkpoint_path):
        print(f"ERROR: Model checkpoint not found at {checkpoint_path}")
        print("Please download the pre-trained model from the link in README.md")
        return
    
    model = load_model(checkpoint_path, device, num_classes=2, dropout=DROPOUT)
    print(f"Model loaded from: {checkpoint_path}")
    
    # Sample tweets for demonstration
    sample_tweets = [
        # Disaster tweets
        "BREAKING: Massive earthquake hits California, buildings collapsed",
        "Forest fire spreading rapidly in Oregon, evacuations ordered",
        "Tsunami warning issued for coastal areas after underwater earthquake",
        
        # Non-disaster tweets
        "Just watched a disaster movie, it was so good!",
        "My kitchen is a war zone after cooking dinner",
        "This concert is absolutely fire! Best night ever!",
    ]
    
    # Classify tweets
    print("\n" + "=" * 70)
    print("CLASSIFICATION RESULTS")
    print("=" * 70)
    
    for tweet in sample_tweets:
        label, confidence = predict_single_tweet(model, tokenizer, tweet, device, MAX_LENGTH)
        icon = "🚨" if label == "DISASTER" else "✅"
        print(f"\n{icon} Tweet: {tweet}")
        print(f"   Prediction: {label} (Confidence: {confidence:.1%})")
    
    # Save results
    results_dir = os.path.join(PROJECT_ROOT, 'results')
    os.makedirs(results_dir, exist_ok=True)
    results_file = os.path.join(results_dir, 'demo_results.txt')
    
    with open(results_file, 'w') as f:
        f.write("Disaster Tweet Classification Results\n")
        f.write("=" * 50 + "\n\n")
        for tweet in sample_tweets:
            label, confidence = predict_single_tweet(model, tokenizer, tweet, device, MAX_LENGTH)
            f.write(f"Tweet: {tweet}\n")
            f.write(f"Prediction: {label} (Confidence: {confidence:.1%})\n\n")
    
    print(f"\n{'=' * 70}")
    print(f"Results saved to: {results_file}")
    print("=" * 70)
    print("\nDemo complete!")


if __name__ == "__main__":
    main()
