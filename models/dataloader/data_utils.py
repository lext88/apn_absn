from common.utils import set_seed
from torch.utils.data import DataLoader
from datasets import load_dataset
from aux_dataloader import get_aux_dataloader

def dataset_builder(args):
    set_seed(args.seed)  # Fix random seed for reproducibility

    if args.dataset == 'mnli':
        dataset = load_dataset('glue', 'mnli')
        texts = [ex['premise'] + " " + ex['hypothesis'] for ex in dataset['train']]
        labels = [ex['label'] for ex in dataset['train']]
        train_loader = get_aux_dataloader(texts, labels, batch_size=args.batch_size, num_workers=args.num_workers)
        
        texts_val = [ex['premise'] + " " + ex['hypothesis'] for ex in dataset['validation_mismatched']]
        labels_val = [ex['label'] for ex in dataset['validation_mismatched']]
        validation_loader = get_aux_dataloader(texts_val, labels_val, batch_size=args.batch_size, num_workers=args.num_workers)
        
        texts_test = [ex['premise'] + " " + ex['hypothesis'] for ex in dataset['test_mismatched']]
        labels_test = [ex['label'] for ex in dataset['test_mismatched']]
        test_loader = get_aux_dataloader(texts_test, labels_test, batch_size=args.batch_size, num_workers=args.num_workers]
        
        return train_loader, validation_loader, test_loader

    elif args.dataset == 'sst2':
        dataset = load_dataset('glue', 'sst2')
        texts = [ex['sentence'] for ex in dataset['train']]
        labels = [ex['label'] for ex in dataset['train']]
        train_loader = get_aux_dataloader(texts, labels, batch_size=args.batch_size, num_workers=args.num_workers)
        
        texts_val = [ex['sentence'] for ex in dataset['validation']]
        labels_val = [ex['label'] for ex in dataset['validation']]
        validation_loader = get_aux_dataloader(texts_val, labels_val, batch_size=args.batch_size, num_workers=args.num_workers)
        
        texts_test = [ex['sentence'] for ex in dataset['test']]
        labels_test = [ex['label'] for ex in dataset['test']]
        test_loader = get_aux_dataloader(texts_test, labels_test, batch_size=args.batch_size, num_workers=args.num_workers]
        
        return train_loader, validation_loader, test_loader

    elif args.dataset == 'qqp':
        dataset = load_dataset('glue', 'qqp')
        texts = [ex['question1'] + " " + ex['question2'] for ex in dataset['train']]
        labels = [ex['label'] for ex in dataset['train']]
        train_loader = get_aux_dataloader(texts, labels, batch_size=args.batch_size, num_workers=args.num_workers)
        
        texts_val = [ex['question1'] + " " + ex['question2'] for ex in dataset['validation']]
        labels_val = [ex['label'] for ex in dataset['validation']]
        validation_loader = get_aux_dataloader(texts_val, labels_val, batch_size=args.batch_size, num_workers=args.num_workers)
        
        texts_test = [ex['question1'] + " " + ex['question2'] for ex in dataset['test']]
        labels_test = [ex['label'] for ex in dataset['test']]
        test_loader = get_aux_dataloader(texts_test, labels_test, batch_size=args.batch_size, num_workers=args.num_workers]
        
        return train_loader, validation_loader, test_loader

    elif args.dataset == 'sts-b':
        dataset = load_dataset('glue', 'sts-b')
        texts = [ex['sentence1'] + " " + ex['sentence2'] for ex in dataset['train']]
        labels = [ex['label'] for ex in dataset['train']]
        train_loader = get_aux_dataloader(texts, labels, batch_size=args.batch_size, num_workers=args.num_workers)
        
        texts_val = [ex['sentence1'] + " " + ex['sentence2'] for ex in dataset['validation']]
        labels_val = [ex['label'] for ex in dataset['validation']]
        validation_loader = get_aux_dataloader(texts_val, labels_val, batch_size=args.batch_size, num_workers=args.num_workers)
        
        texts_test = [ex['sentence1'] + " " + ex['sentence2'] for ex in dataset['test']]
        labels_test = [ex['label'] for ex in dataset['test']]
        test_loader = get_aux_dataloader(texts_test, labels_test, batch_size=args.batch_size, num_workers=args.num_workers]
        
        return train_loader, validation_loader, test_loader

    elif args.dataset == 'rte':
        dataset = load_dataset('glue', 'rte')
        texts = [ex['premise'] + " " + ex['hypothesis'] for ex in dataset['train']]
        labels = [ex['label'] for ex in dataset['train']]
        train_loader = get_aux_dataloader(texts, labels, batch_size=args.batch_size, num_workers=args.num_workers)
        
        texts_val = [ex['premise'] + " " + ex['hypothesis'] for ex in dataset['validation']]
        labels_val = [ex['label'] for ex in dataset['validation']]
        validation_loader = get_aux_dataloader(texts_val, labels_val, batch_size=args.batch_size, num_workers=args.num_workers)
        
        texts_test = [ex['premise'] + " " + ex['hypothesis'] for ex in dataset['test']]
        labels_test = [ex['label'] for ex in dataset['test']]
        test_loader = get_aux_dataloader(texts_test, labels_test, batch_size=args.batch_size, num_workers=args.num_workers]
        
        return train_loader, validation_loader, test_loader

    elif args.dataset == 'qnli':
        dataset = load_dataset('glue', 'qnli')
        texts = [ex['question'] + " " + ex['sentence'] for ex in dataset['train']]
        labels = [ex['label'] for ex in dataset['train']]
        train_loader = get_aux_dataloader(texts, labels, batch_size=args.batch_size, num_workers=args.num_workers)
        
        texts_val = [ex['question'] + " " + ex['sentence'] for ex in dataset['validation']]
        labels_val = [ex['label'] for ex in dataset['validation']]
        validation_loader = get_aux_dataloader(texts_val, labels_val, batch_size=args.batch_size, num_workers=args.num_workers)
        
        texts_test = [ex['question'] + " " + ex['sentence'] for ex in dataset['test']]
        labels_test = [ex['label'] for ex in dataset['test']]
        test_loader = get_aux_dataloader(texts_test, labels_test, batch_size=args.batch_size, num_workers=args.num_workers]
        
        return train_loader, validation_loader, test_loader

    elif args.dataset == 'wnli':
        dataset = load_dataset('glue', 'wnli')
        texts = [ex['sentence1'] + " " + ex['sentence2'] for ex in dataset['train']]
        labels = [ex['label'] for ex in dataset['train']]
        train_loader = get_aux_dataloader(texts, labels, batch_size=args.batch_size, num_workers=args.num_workers)
        
        texts_val = [ex['sentence1'] + " " + ex['sentence2'] for ex in dataset['validation']]
        labels_val = [ex['label'] for ex in dataset['validation']]
        validation_loader = get_aux_dataloader(texts_val, labels_val, batch_size=args.batch_size, num_workers=args.num_workers)
        
        texts_test = [ex['sentence1'] + " " + ex['sentence2'] for ex in dataset['test']]
        labels_test = [ex['label'] for ex in dataset['test']]
        test_loader = get_aux_dataloader(texts_test, labels_test, batch_size=args.batch_size, num_workers=args.num_workers]
        
        return train_loader, validation_loader, test_loader

    else:
        raise ValueError('Unknown Dataset')
