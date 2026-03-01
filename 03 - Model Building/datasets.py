from torch.utils.data import Dataset
import torch

# Base dataset (for RNN, Logistic Regression)
class DialogueDataset(Dataset):
    """Basic dataset without special tokens."""
    def __init__(self, tokens_list, labels):
        self.tokens_list = tokens_list
        self.labels = labels
    
    def __len__(self):
        return len(self.labels)
    
    def __getitem__(self, idx):
        return {
            'text': torch.tensor(self.tokens_list[idx], dtype=torch.long),
            'label': torch.tensor(self.labels[idx], dtype=torch.long),
            'length': len(self.tokens_list[idx])
        }


# BERT-specific dataset (adds CLS token)
class BERTDialogueDataset(Dataset):
    """Dataset with [CLS] token for BERT-style models."""
    def __init__(self, tokens_list, labels, vocab, max_len):
        self.tokens_list = tokens_list
        self.labels = labels
        self.vocab = vocab
        self.max_len = max_len
        self.cls_idx = vocab.word2idx['<cls>']
    
    def __len__(self):
        return len(self.labels)
    
    def __getitem__(self, idx):
        tokens = self.tokens_list[idx]
        label = self.labels[idx]
        
        # Encode and prepend [CLS]
        token_ids = self.vocab.encode(tokens)
        token_ids = [self.cls_idx] + token_ids[:self.max_len - 1]
        
        return {
            'input_ids': token_ids,
            'label': label,
            'length': len(token_ids)
        }


# Collate functions
def collate_fn_rnn(batch):
    """Collate for RNN (sorts by length for pack_padded_sequence)."""
    batch = sorted(batch, key=lambda x: x['length'], reverse=True)
    max_len = batch[0]['length']
    
    texts, labels, lengths = [], [], []
    for item in batch:
        text = item['text']
        padded = torch.cat([text, torch.zeros(max_len - len(text), dtype=torch.long)])
        texts.append(padded)
        labels.append(item['label'])
        lengths.append(item['length'])
    
    return {
        'text': torch.stack(texts),
        'label': torch.stack(labels),
        'length': torch.tensor(lengths, dtype=torch.long)
    }


def collate_fn_bert(batch):
    """Collate for BERT (with attention masks)."""
    input_ids = [item['input_ids'] for item in batch]
    labels = [item['label'] for item in batch]
    lengths = [item['length'] for item in batch]
    
    max_len = max(lengths)
    padded_input_ids = []
    attention_masks = []
    
    for ids, length in zip(input_ids, lengths):
        padding_length = max_len - length
        padded_ids = ids + [0] * padding_length
        padded_input_ids.append(padded_ids)
        
        mask = [1] * length + [0] * padding_length
        attention_masks.append(mask)
    
    return {
        'input_ids': torch.tensor(padded_input_ids, dtype=torch.long),
        'attention_mask': torch.tensor(attention_masks, dtype=torch.long),
        'labels': torch.tensor(labels, dtype=torch.long),
        'lengths': torch.tensor(lengths, dtype=torch.long)
    }


def collate_fn_logistic(batch):
    """Collate for logistic regression (simple padding)."""
    labels = torch.tensor([item['label'] for item in batch])
    texts = [item['text'] for item in batch]
    
    max_len = max(len(text) for text in texts)
    padded_texts = torch.zeros(len(texts), max_len, dtype=torch.long)
    
    for i, text in enumerate(texts):
        padded_texts[i, :len(text)] = text
    
    return padded_texts, labels
