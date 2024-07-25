from common.utils import set_seed
from transformers import BertTokenizer
from torch.utils.data import DataLoader

def dataset_builder(args):
    set_seed(args.seed)  # Fix random seed for reproducibility

    if args.dataset == 'miniImageNet':
        from models.dataloader.mini_imagenet import MiniImageNet as Dataset
        # Return DataLoader
        dataset = Dataset(args)
        return DataLoader(dataset, batch_size=args.batch_size, shuffle=True, num_workers=args.num_workers)
    
    elif args.dataset == 'cub':
        from models.dataloader.cub import CUB as Dataset
        # Return DataLoader
        dataset = Dataset(args)
        return DataLoader(dataset, batch_size=args.batch_size, shuffle=True, num_workers=args.num_workers)
    
    elif args.dataset == 'tieredImageNet':
        from models.dataloader.tiered_imagenet import tieredImageNet as Dataset
        # Return DataLoader
        dataset = Dataset(args)
        return DataLoader(dataset, batch_size=args.batch_size, shuffle=True, num_workers=args.num_workers)
    
    elif args.dataset == 'CIFAR-FS':
        from models.dataloader.cifar_fs import DatasetLoader as Dataset
        # Return DataLoader
        dataset = Dataset(args)
        return DataLoader(dataset, batch_size=args.batch_size, shuffle=True, num_workers=args.num_workers)
    
    elif args.dataset in ['mnli', 'sst2', 'qqp', 'sts-b', 'rte', 'qnli', 'wnli']:
        from datasets import load_dataset
        
        # Define a tokenizer for GLUE tasks
        tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
        
        def preprocess_function(examples):
            # Tokenize and prepare data for BERT
            return tokenizer(examples['sentence1'], examples.get('sentence2', None), truncation=True, padding='max_length', max_length=128)
        
        # Load dataset from HuggingFace datasets
        dataset = load_dataset('glue', args.dataset)
        
        # Preprocess and prepare dataset
        dataset = dataset.map(preprocess_function, batched=True)
        dataset.set_format(type='torch', columns=['input_ids', 'attention_mask', 'label'])
        
        # Create DataLoaders
        train_loader = DataLoader(dataset['train'], batch_size=args.batch_size, shuffle=True, num_workers=args.num_workers)
        validation_loader = DataLoader(dataset['validation'], batch_size=args.batch_size, shuffle=False, num_workers=args.num_workers)
        test_loader = DataLoader(dataset['test'], batch_size=args.batch_size, shuffle=False, num_workers=args.num_workers)
        
        return train_loader, validation_loader, test_loader

    else:
        raise ValueError('Unknown Dataset')
