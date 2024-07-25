"""For GLUE tasks like Multi-Genre Natural Language Inference (MNLI), you would prepare the texts and labels according to the specific task format. 
For MNLI, you'll likely need to handle pairs of sentences and their labels. Here’s an example snippet for preparing data for MNLI:

from datasets import load_dataset

def load_mnli_data():
    dataset = load_dataset('glue', 'mnli')
    train_texts = dataset['train']['sentence1'] + dataset['train']['sentence2']
    train_labels = dataset['train']['label']
    return train_texts, train_labels

# Example usage
texts, labels = load_mnli_data()
dataloader = get_aux_dataloader(texts, labels, tokenizer_name='bert-base-uncased', max_len=128, batch_size=32, num_workers=4)

This approach prepares the data for training your model, ensuring that it’s compatible with your text-based neural network and the GLUE benchmark tasks.

"""


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

def get_text_dataloader(texts, labels, tokenizer_name='bert-base-uncased', max_len=128, batch_size=32, num_workers=4, n_cls=5, n_per=5):
    # Create dataset and sampler
    dataset = TextPreprocessor(texts, labels, tokenizer_name, max_len)
    sampler = TextCategoriesSampler(labels, n_batch=(len(labels) // batch_size), n_cls=n_cls, n_per=n_per)

    # Create DataLoader
    dataloader = DataLoader(dataset, batch_size=batch_size, sampler=sampler, num_workers=num_workers, pin_memory=True)
    return dataloader
