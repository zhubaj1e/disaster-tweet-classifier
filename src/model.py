"""
Disaster Tweet Classifier Model

This module defines the DistilBERT-based classifier for disaster tweet classification.
"""

import torch
import torch.nn as nn
from transformers import DistilBertModel


class DisasterTweetClassifier(nn.Module):
    """
    DistilBERT-based classifier with custom classification head.
    Uses [CLS] token representation for sequence classification.
    
    Architecture:
        Input Tweet → Tokenizer → DistilBERT → [CLS] token → Custom Head → Classification
                                                    ↓
                            768-dim → Dropout → 256-dim → ReLU → Dropout → 2-dim
    
    Args:
        num_classes (int): Number of output classes (default: 2 for binary classification)
        dropout (float): Dropout probability for regularization (default: 0.3)
    """
    
    def __init__(self, num_classes: int = 2, dropout: float = 0.3):
        super(DisasterTweetClassifier, self).__init__()
        
        # Load pre-trained DistilBERT
        self.bert = DistilBertModel.from_pretrained('distilbert-base-uncased')
        self.hidden_size = self.bert.config.hidden_size  # 768
        
        # Custom classification head
        self.classifier = nn.Sequential(
            nn.Dropout(dropout),
            nn.Linear(self.hidden_size, 256),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(256, num_classes)
        )
        
    def forward(self, input_ids: torch.Tensor, attention_mask: torch.Tensor) -> torch.Tensor:
        """
        Forward pass of the model.
        
        Args:
            input_ids: Tokenized input tensor of shape (batch_size, seq_length)
            attention_mask: Attention mask tensor of shape (batch_size, seq_length)
            
        Returns:
            Logits tensor of shape (batch_size, num_classes)
        """
        # Get DistilBERT outputs
        outputs = self.bert(input_ids=input_ids, attention_mask=attention_mask)
        
        # Extract [CLS] token representation (first token)
        cls_output = outputs.last_hidden_state[:, 0, :]
        
        # Pass through classification head
        return self.classifier(cls_output)
    
    def predict(self, input_ids: torch.Tensor, attention_mask: torch.Tensor) -> tuple:
        """
        Make predictions with confidence scores.
        
        Args:
            input_ids: Tokenized input tensor
            attention_mask: Attention mask tensor
            
        Returns:
            Tuple of (predictions, confidence_scores)
        """
        self.eval()
        with torch.no_grad():
            outputs = self.forward(input_ids, attention_mask)
            probs = torch.softmax(outputs, dim=1)
            predictions = torch.argmax(outputs, dim=1)
            confidence = probs.gather(1, predictions.unsqueeze(1)).squeeze(1)
        return predictions, confidence


def load_model(checkpoint_path: str, device: torch.device, num_classes: int = 2, 
               dropout: float = 0.3) -> DisasterTweetClassifier:
    """
    Load a trained model from checkpoint.
    
    Args:
        checkpoint_path: Path to the saved model checkpoint
        device: Device to load the model onto
        num_classes: Number of output classes
        dropout: Dropout probability
        
    Returns:
        Loaded DisasterTweetClassifier model
    """
    model = DisasterTweetClassifier(num_classes=num_classes, dropout=dropout)
    model.load_state_dict(torch.load(checkpoint_path, map_location=device))
    model.to(device)
    model.eval()
    return model
