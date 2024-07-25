from __future__ import print_function

import torch
from torch.utils.data import Dataset, DataLoader
from transformers import BertTokenizer

class TextPreprocessor(Dataset):
    def __init__(self, texts, labels, tokenizer_name='bert-base-uncased', max_len=128):
        self.texts = texts
        self.labels = labels
        self.tokenizer = BertTokenizer.from_pretrained(tokenizer_name)
        self.max_len = max_len

    def __len__(self):
        return len(self.texts)

    def __getitem__(self, idx):
        text = self.texts[idx]
        label = self.labels[idx]

        encoding = self.tokenizer.encode_plus(
            text,
            add_special_tokens=True,
            max_length=self.max_len,
            padding='max_length',
            truncation=True,
            return_tensors='pt'
        )

        input_ids = encoding['input_ids'].flatten()
        attention_mask = encoding['attention_mask'].flatten()

        return {
            'input_ids': input_ids,
            'attention_mask': attention_mask,
            'labels': torch.tensor(label, dtype=torch.long)
        }

def get_aux_dataloader(texts, labels, tokenizer_name='bert-base-uncased', max_len=128, batch_size=32, num_workers=4):
    dataset = TextPreprocessor(texts, labels, tokenizer_name, max_len)
    return DataLoader(dataset, batch_size=batch_size, shuffle=True, drop_last=True, num_workers=num_workers, pin_memory=True)
