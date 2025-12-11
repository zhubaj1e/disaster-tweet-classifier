# Disaster Tweet Classifier

A DistilBERT-based classifier for identifying disaster-related tweets. Developed for EEP 596 Final Project.

## Project Overview

Classifies tweets as disaster-related or not using a fine-tuned DistilBERT transformer. Achieves **0.788 F1 score** and **81.6% accuracy**, outperforming TF-IDF baselines.

## Setup Instructions

```bash
# 1. Clone and enter repository
git clone https://github.com/YOUR_USERNAME/disaster-tweet-classifier.git
cd disaster-tweet-classifier

# 2. Create virtual environment
python -m venv venv
venv\Scripts\activate  # Windows
# source venv/bin/activate  # macOS/Linux

# 3. Install dependencies
pip install -r requirements.txt
```

## How to Run

```bash
python demo/demo.py
```

Or open `demo/demo.ipynb` in Jupyter/VS Code.

## Expected Output

```
Tweet: BREAKING: Massive earthquake hits California, buildings collapsed
   Prediction: DISASTER (Confidence: 99.6%)

Tweet: Just watched a disaster movie, it was so good!
   Prediction: NOT DISASTER (Confidence: 89.7%)
```

Results are saved to `results/demo_results.txt`.

## Pre-trained Model

The trained model is included in the repository at `checkpoints/disaster_tweet_classifier.pt`. No external download required.

## Hyperparameters

See `src/config.py` for all training settings. Key parameters:
- Learning Rate: 3e-5
- Batch Size: 32
- Dropout: 0.3
- Epochs: 5

## Acknowledgments

- **Dataset**: [Kaggle NLP with Disaster Tweets](https://www.kaggle.com/competitions/nlp-getting-started)
- **Base Model**: [DistilBERT](https://huggingface.co/distilbert-base-uncased) by Hugging Face
- **Course**: EEP 596 - Introduction to Deep Learning
