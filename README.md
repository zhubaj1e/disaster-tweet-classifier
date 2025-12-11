# Disaster Tweet Classifier

A DistilBERT-based classifier for identifying disaster-related tweets. Developed for EEP 596 Final Project.

## Project Overview

Classifies tweets as disaster-related or not using a fine-tuned DistilBERT transformer. Achieves **0.788 F1 score** and **81.6% accuracy**, outperforming TF-IDF baselines.

## Setup Instructions

```bash
# 1. Clone and enter repository
git clone https://github.com/zhubaj1e/disaster-tweet-classifier.git
cd disaster-tweet-classifier

# 2. Create virtual environment
python -m venv venv
venv\Scripts\activate  # Windows
# source venv/bin/activate  # macOS/Linux

# 3. Install dependencies
pip install -r requirements.txt
```

## How to Run

1. Download the pre-trained model using the Google Drive link above and place it at `checkpoints/disaster_tweet_classifier.pt`.

2. Run the demo script:

```bash
python demo/demo.py
```

Or open `demo/demo.ipynb` in Jupyter/VS Code and run the cells. The demo uses the model checkpoint from `checkpoints/disaster_tweet_classifier.pt` and produces output saved to `results/`.

## Expected Output

```
Tweet: BREAKING: Massive earthquake hits California, buildings collapsed
   Prediction: DISASTER (Confidence: 99.6%)

Tweet: Just watched a disaster movie, it was so good!
   Prediction: NOT DISASTER (Confidence: 89.7%)
```

Results are saved to `results/demo_results.txt`.

## Pre-trained Model

The trained model is available from the Google Drive link below. Download and place the file in the `checkpoints` folder (create it if necessary) using the exact filename `disaster_tweet_classifier.pt`.

**Google Drive (Model Download):** https://drive.google.com/file/d/17N7EZAiJb8BnX72CMjvcKRy0wgx-bvC_/view?usp=drive_link

Example (Windows PowerShell):
```powershell
# Create checkpoints folder and download manually via browser, then move the file to the folder
mkdir checkpoints
# Place downloaded file in checkpoints/disaster_tweet_classifier.pt
```

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
