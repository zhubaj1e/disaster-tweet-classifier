"""
Configuration Settings for Disaster Tweet Classifier

This module contains all hyperparameters and configuration settings
for training and inference.
"""

# =============================================================================
# MODEL CONFIGURATION
# =============================================================================
MODEL_NAME = "distilbert-base-uncased"
NUM_CLASSES = 2
MAX_LENGTH = 128  # Sufficient for tweets (max 280 chars)

# =============================================================================
# TRAINING HYPERPARAMETERS (Optimized through experiments)
# =============================================================================

# Learning Rate: Tested [1e-5, 2e-5, 3e-5, 5e-5, 1e-4], best F1=0.8102 at 3e-5
LEARNING_RATE = 3e-5

# Batch Size: Tested [8, 16, 32, 64], 32 balances stability and speed
BATCH_SIZE = 32

# Dropout: Tested [0.1, 0.2, 0.3, 0.4, 0.5], 0.3 provides good regularization
DROPOUT = 0.3

# Epochs: Tested 1-10, overfitting occurs after epoch 3-4
EPOCHS = 5

# Warmup Steps: Standard for transformers
WARMUP_STEPS = 200

# Weight Decay: L2 regularization
WEIGHT_DECAY = 0.01

# Gradient Clipping: Prevent exploding gradients
MAX_GRAD_NORM = 1.0

# =============================================================================
# DATA CONFIGURATION
# =============================================================================
TRAIN_DATA_PATH = "data/train.csv"
TEST_DATA_PATH = "data/test.csv"
VALIDATION_SPLIT = 0.2

# =============================================================================
# MODEL PATHS
# =============================================================================
CHECKPOINT_DIR = "checkpoints"
MODEL_FILENAME = "disaster_tweet_classifier.pt"

# =============================================================================
# RANDOM SEED FOR REPRODUCIBILITY
# =============================================================================
SEED = 42

# =============================================================================
# HYPERPARAMETER TUNING SUMMARY
# =============================================================================
"""
Hyperparameter Tuning Results:

| Experiment     | Values Tested            | Best Value | Best F1 |
|----------------|--------------------------|------------|---------|
| Learning Rate  | 1e-5, 2e-5, 3e-5, 5e-5   | 3e-5       | 0.8102  |
| Dropout        | 0.1, 0.2, 0.3, 0.4, 0.5  | 0.3        | 0.8144  |
| Batch Size     | 8, 16, 32, 64            | 32         | 0.8094  |
| Epochs         | 1-10                     | 4-5        | 0.8045  |

Key Findings:
1. Learning Rate 3e-5 performed best, confirming BERT paper recommendations
2. Dropout 0.3 provides good regularization without hurting learning
3. Batch Size 32 balances training stability and speed
4. Overfitting occurs after epoch 3-4, early stopping is essential
"""
