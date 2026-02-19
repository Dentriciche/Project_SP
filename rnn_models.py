"""
Recurrent Neural Network Models for South Park Character Classification

This module contains three recurrent architectures:
- RNNClassifier: Simple RNN
- LSTMClassifier: Long Short-Term Memory
- GRUClassifier: Gated Recurrent Unit
"""

import torch
import torch.nn as nn
from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence


class RNNClassifier(nn.Module):
    """Simple RNN for character classification."""
    
    def __init__(self, vocab_size, embedding_dim, hidden_dim, num_classes, 
                 num_layers=1, dropout_rate=0.5, bidirectional=False):
        super(RNNClassifier, self).__init__()
        
        self.hidden_dim = hidden_dim
        self.num_layers = num_layers
        self.bidirectional = bidirectional
        self.num_directions = 2 if bidirectional else 1
        
        # Embedding layer
        self.embedding = nn.Embedding(vocab_size, embedding_dim, padding_idx=0)
        
        # RNN layer
        self.rnn = nn.RNN(
            embedding_dim, 
            hidden_dim, 
            num_layers=num_layers,
            batch_first=True,
            dropout=dropout_rate if num_layers > 1 else 0,
            bidirectional=bidirectional
        )
        
        # Dropout
        self.dropout = nn.Dropout(dropout_rate)
        
        # Output layer
        self.fc = nn.Linear(hidden_dim * self.num_directions, num_classes)
    
    def forward(self, text, lengths):
        # text: (batch_size, seq_len)
        # lengths: (batch_size,)
        
        # Embed
        embedded = self.embedding(text)
        
        # Pack padded sequences for efficiency
        packed = pack_padded_sequence(embedded, lengths.cpu(), batch_first=True, enforce_sorted=True)
        
        # RNN forward pass
        packed_output, hidden = self.rnn(packed)
        
        # Get final hidden state
        if self.bidirectional:
            # Concatenate forward and backward hidden states from last layer
            hidden = torch.cat((hidden[-2], hidden[-1]), dim=1)
        else:
            hidden = hidden[-1]
        
        # Dropout and classify
        hidden = self.dropout(hidden)
        output = self.fc(hidden)
        
        return output


class LSTMClassifier(nn.Module):
    """LSTM for character classification."""
    
    def __init__(self, vocab_size, embedding_dim, hidden_dim, num_classes, 
                 num_layers=1, dropout_rate=0.5, bidirectional=False):
        super(LSTMClassifier, self).__init__()
        
        self.hidden_dim = hidden_dim
        self.num_layers = num_layers
        self.bidirectional = bidirectional
        self.num_directions = 2 if bidirectional else 1
        
        # Embedding layer
        self.embedding = nn.Embedding(vocab_size, embedding_dim, padding_idx=0)
        
        # LSTM layer
        self.lstm = nn.LSTM(
            embedding_dim, 
            hidden_dim, 
            num_layers=num_layers,
            batch_first=True,
            dropout=dropout_rate if num_layers > 1 else 0,
            bidirectional=bidirectional
        )
        
        # Dropout
        self.dropout = nn.Dropout(dropout_rate)
        
        # Output layer
        self.fc = nn.Linear(hidden_dim * self.num_directions, num_classes)
    
    def forward(self, text, lengths):
        # Embed
        embedded = self.embedding(text)
        
        # Pack padded sequences
        packed = pack_padded_sequence(embedded, lengths.cpu(), batch_first=True, enforce_sorted=True)
        
        # LSTM forward pass
        packed_output, (hidden, cell) = self.lstm(packed)
        
        # Get final hidden state
        if self.bidirectional:
            hidden = torch.cat((hidden[-2], hidden[-1]), dim=1)
        else:
            hidden = hidden[-1]
        
        # Dropout and classify
        hidden = self.dropout(hidden)
        output = self.fc(hidden)
        
        return output


class GRUClassifier(nn.Module):
    """GRU for character classification."""
    
    def __init__(self, vocab_size, embedding_dim, hidden_dim, num_classes, 
                 num_layers=1, dropout_rate=0.5, bidirectional=False):
        super(GRUClassifier, self).__init__()
        
        self.hidden_dim = hidden_dim
        self.num_layers = num_layers
        self.bidirectional = bidirectional
        self.num_directions = 2 if bidirectional else 1
        
        # Embedding layer
        self.embedding = nn.Embedding(vocab_size, embedding_dim, padding_idx=0)
        
        # GRU layer
        self.gru = nn.GRU(
            embedding_dim, 
            hidden_dim, 
            num_layers=num_layers,
            batch_first=True,
            dropout=dropout_rate if num_layers > 1 else 0,
            bidirectional=bidirectional
        )
        
        # Dropout
        self.dropout = nn.Dropout(dropout_rate)
        
        # Output layer
        self.fc = nn.Linear(hidden_dim * self.num_directions, num_classes)
    
    def forward(self, text, lengths):
        # Embed
        embedded = self.embedding(text)
        
        # Pack padded sequences
        packed = pack_padded_sequence(embedded, lengths.cpu(), batch_first=True, enforce_sorted=True)
        
        # GRU forward pass
        packed_output, hidden = self.gru(packed)
        
        # Get final hidden state
        if self.bidirectional:
            hidden = torch.cat((hidden[-2], hidden[-1]), dim=1)
        else:
            hidden = hidden[-1]
        
        # Dropout and classify
        hidden = self.dropout(hidden)
        output = self.fc(hidden)
        
        return output

