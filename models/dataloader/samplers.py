import torch
import numpy as np

class TextCategoriesSampler:
    def __init__(self, labels, n_batch, n_cls, n_per):
        self.n_batch = n_batch  # Number of batches to sample
        self.n_cls = n_cls  # Number of classes per batch
        self.n_per = n_per  # Number of samples per class

        # Convert labels to numpy array for processing
        labels = np.array(labels)
        
        # Store indices of each class
        self.class_indices = []
        for i in range(max(labels) + 1):
            indices = np.argwhere(labels == i).reshape(-1)  # Indices of data in class i
            indices = torch.from_numpy(indices)
            self.class_indices.append(indices)

    def __len__(self):
        return self.n_batch
    
    def __iter__(self):
        for _ in range(self.n_batch):
            batch_indices = []
            # Randomly sample class indices
            classes = torch.randperm(len(self.class_indices))[:self.n_cls]
            for cls in classes:
                indices = self.class_indices[cls]  # Indices of the selected class
                sampled_indices = torch.randperm(len(indices))[:self.n_per]  # Sample indices for this class
                batch_indices.append(indices[sampled_indices])
            
            # Flatten and yield the batch
            batch_indices = torch.stack(batch_indices).t().reshape(-1)
            yield batch_indices
