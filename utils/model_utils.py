from typing import List, Dict, Tuple
from collections import Counter
import torch
from torch.utils.data import DataLoader
import re
from config import configs

def preprocess_text(text: str) -> List[str]:
    """Clean and tokenize text."""
    text = text.lower()
    text = re.sub(r'[^a-z\s]', '', text)
    tokens = text.split()
    return tokens


def create_dataloaders(train_tokens, train_labels, val_tokens, val_labels,
                       test_tokens, test_labels, vocab, model_type,
                       max_seq_len=256, batch_size=64, configs=configs):
    """
    Universal dataloader creator for all model types.
    
    Usage:
        # BERT
        loaders = create_dataloaders(..., model_type='bert')
        
        # RNN
        loaders = create_dataloaders(..., model_type='lstm')
    """
    # Configuration mapping
    
    if model_type not in configs:
        raise ValueError(f"Unknown model_type: {model_type}. Choose from {list(configs.keys())}")
    
    config = configs[model_type]
    
    # Create datasets with appropriate arguments
    dataset_kwargs = {}
    if config['needs_vocab']:
        dataset_kwargs['vocab'] = vocab
    if config['needs_max_len']:
        dataset_kwargs['max_len'] = max_seq_len
    
    train_dataset = config['dataset'](train_tokens, train_labels, **dataset_kwargs)
    val_dataset = config['dataset'](val_tokens, val_labels, **dataset_kwargs)
    test_dataset = config['dataset'](test_tokens, test_labels, **dataset_kwargs)
    
    # Create dataloaders
    train_loader = DataLoader(train_dataset, batch_size=batch_size, 
                             shuffle=True, collate_fn=config['collate'])
    val_loader = DataLoader(val_dataset, batch_size=batch_size, 
                           shuffle=False, collate_fn=config['collate'])
    test_loader = DataLoader(test_dataset, batch_size=batch_size, 
                            shuffle=False, collate_fn=config['collate'])
    
    return train_loader, val_loader, test_loader

class Vocabulary:
    """Vocabulary with special tokens for BERT-style models."""
    
    def __init__(self, min_freq=1, max_size=None, special_tokens: List[str] = ['<pad>', '<unk>', '<cls>', '<sep>']):
        self.min_freq = min_freq
        self.max_size = max_size
        self.word2idx = {token: idx for idx, token in enumerate(special_tokens)}
        self.idx2word = {idx: token for idx, token in enumerate(special_tokens)}
        self.word_counts = Counter()
    
    def build_vocab(self, texts: List[List[str]]):
        """Build vocabulary from tokenized texts."""
        for tokens in texts:
            self.word_counts.update(tokens)
        
        # Filter by frequency
        filtered_words = {w: c for w, c in self.word_counts.items() if c >= self.min_freq}
        
        # Sort by frequency
        sorted_words = sorted(filtered_words.items(), key=lambda x: x[1], reverse=True)
        
        # Limit vocabulary size
        if self.max_size:
            sorted_words = sorted_words[:self.max_size - 4]  # Reserve space for special tokens
        
        # Add to vocabulary
        for word, _ in sorted_words:
            if word not in self.word2idx:
                idx = len(self.word2idx)
                self.word2idx[word] = idx
                self.idx2word[idx] = word
    
    def encode(self, tokens: List[str]) -> List[int]:
        """Convert tokens to indices."""
        return [self.word2idx.get(token, self.word2idx['<unk>']) for token in tokens]
    
    def decode(self, indices: List[int]) -> List[str]:
        """Convert indices to tokens."""
        return [self.idx2word.get(idx, '<unk>') for idx in indices]
    
    def __len__(self):
        return len(self.word2idx)


