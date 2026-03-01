from datasets import BERTDialogueDataset, DialogueDataset, collate_fn_bert, collate_fn_rnn, collate_fn_logistic
configs = {
        'bert': {
            'dataset': BERTDialogueDataset,
            'collate': collate_fn_bert,
            'needs_vocab': True,
            'needs_max_len': True
        },
        'distilbert': {
            'dataset': BERTDialogueDataset,
            'collate': collate_fn_bert,
            'needs_vocab': True,
            'needs_max_len': True
        },
        'rnn': {
            'dataset': DialogueDataset,
            'collate': collate_fn_rnn,
            'needs_vocab': False,
            'needs_max_len': False
        },
        'lstm': {
            'dataset': DialogueDataset,
            'collate': collate_fn_rnn,
            'needs_vocab': False,
            'needs_max_len': False
        },
        'gru': {
            'dataset': DialogueDataset,
            'collate': collate_fn_rnn,
            'needs_vocab': False,
            'needs_max_len': False
        },
        'logistic': {
            'dataset': DialogueDataset,
            'collate': collate_fn_logistic,
            'needs_vocab': False,
            'needs_max_len': False
        }
    }
    