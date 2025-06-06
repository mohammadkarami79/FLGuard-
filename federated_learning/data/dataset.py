import torch
from torch.utils.data import Dataset, DataLoader
from torchvision import datasets, transforms
import random
import numpy as np
from federated_learning.config.config import *
from federated_learning.data.alzheimer_dataset import load_alzheimer_dataset, download_alzheimer_dataset
from federated_learning.data.cifar_dataset import load_cifar10_dataset

class LabelFlippingDataset(Dataset):
    """
    Dataset wrapper that implements label flipping attack.
    Flips each label l to (num_classes - l - 1).
    
    For example, in a 10-class dataset:
    0 -> 9, 1 -> 8, 2 -> 7, etc.
    """
    def __init__(self, dataset, num_classes):
        self.dataset = dataset
        self.num_classes = num_classes

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, idx):
        data, target = self.dataset[idx]
        flipped_target = self.num_classes - target - 1
        return data, flipped_target

class BackdoorDataset(Dataset):
    """
    Dataset wrapper that implements backdoor attack.
    Adds a trigger pattern to the data and sets all labels to a target label (0 by default).
    
    This creates poisoned examples that the model will associate with the target label.
    """
    def __init__(self, dataset, num_classes, target_label=0):
        self.dataset = dataset
        self.num_classes = num_classes
        self.target_label = target_label
        self.trigger = self.create_trigger()

    def create_trigger(self):
        """
        Creates a trigger pattern to add to images.
        Currently creates a small white square in the bottom right corner.
        """
        trigger = torch.zeros((1, 28, 28))
        trigger[:, 24:, 24:] = 1.0
        return trigger

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, idx):
        data, _ = self.dataset[idx]  # Original target is ignored
        # Add trigger to the image
        data = data + self.trigger
        data = torch.clamp(data, 0, 1)  # Ensure pixel values stay in valid range
        # Set to target label
        return data, self.target_label

class AdaptiveAttackDataset(Dataset):
    """
    Dataset wrapper for adaptive attacks.
    
    Can be extended with specific modifications to evade defensive measures.
    Currently acts as a pass-through wrapper.
    """
    def __init__(self, dataset):
        self.dataset = dataset

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, idx):
        data, target = self.dataset[idx]
        # Can implement sophisticated adaptive data poisoning here
        return data, target

class MinMaxAttackDataset(Dataset):
    """
    Dataset wrapper that implements the min-max attack as described in FLTrust paper.
    
    This attack tries to maximize negative impact on accuracy while minimizing 
    the chance of being detected. It specifically targets the most confusing class
    for each true class.
    """
    def __init__(self, dataset, num_classes):
        self.dataset = dataset
        self.num_classes = num_classes
        # Create a confusion matrix to determine most confusing classes
        # For simplicity, we use a predefined pattern: target = (original + 1) % num_classes
        self.confusion_map = [(i + 1) % num_classes for i in range(num_classes)]

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, idx):
        data, target = self.dataset[idx]
        # Map to the confusing class
        confused_target = self.confusion_map[target]
        return data, confused_target

class MinSumAttackDataset(Dataset):
    """
    Dataset wrapper that implements the min-sum attack.
    
    This attack aims to minimize the sum of cosine similarities with benign updates.
    In practice, it creates a consistent but subtle pattern across all samples
    to gradually push the model in a specific wrong direction.
    """
    def __init__(self, dataset, num_classes):
        self.dataset = dataset
        self.num_classes = num_classes
        # Create a probability map - instead of just flipping, we introduce uncertainty
        self.prob_map = torch.zeros(num_classes, num_classes)
        for i in range(num_classes):
            # Invert the probability distribution - highest prob for opposite class
            for j in range(num_classes):
                if i == j:
                    self.prob_map[i, j] = 0.1  # Small prob for correct class
                else:
                    # Higher prob for classes further away (in circular sense)
                    distance = min((j - i) % num_classes, (i - j) % num_classes)
                    self.prob_map[i, j] = 1.0 / distance if distance > 0 else 0.1
            # Normalize to create a probability distribution
            self.prob_map[i] = self.prob_map[i] / self.prob_map[i].sum()

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, idx):
        data, target = self.dataset[idx]
        # Sample from the probability distribution for this class
        new_target = torch.multinomial(self.prob_map[target], 1).item()
        return data, new_target

class AlternatingAttackDataset(Dataset):
    """
    Dataset wrapper that implements the alternating attack.
    
    This attack alternates between correct and incorrect labels in a pattern
    that's difficult to detect but creates systematic bias.
    """
    def __init__(self, dataset, num_classes):
        self.dataset = dataset
        self.num_classes = num_classes
        # Create alternating patterns for different samples
        self.alternating_offset = np.random.randint(0, num_classes, size=len(dataset))

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, idx):
        data, target = self.dataset[idx]
        # Use sample index to determine if we should alter the label
        if idx % 2 == 0:
            # Even indices: use offset to create deterministic but varied pattern
            altered_target = (target + self.alternating_offset[idx]) % self.num_classes
            return data, altered_target
        else:
            # Odd indices: keep original
            return data, target

class TargetedAttackDataset(Dataset):
    """
    Dataset wrapper that implements a targeted attack.
    
    This attack focuses on a specific subset of the data (e.g., a particular class)
    and makes targeted modifications to mislead the model specifically on that subset.
    """
    def __init__(self, dataset, num_classes, target_class=0, target_output=1):
        self.dataset = dataset
        self.num_classes = num_classes
        self.target_class = target_class  # The class to attack
        self.target_output = target_output  # The incorrect output to produce

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, idx):
        data, target = self.dataset[idx]
        # Only modify samples from the target class
        if target == self.target_class:
            return data, self.target_output
        return data, target

class GradientInversionAttackDataset(Dataset):
    """
    Dataset wrapper that implements a gradient inversion attack.
    
    This sophisticated attack creates a pattern of label modifications
    that result in gradient updates that have varying effects on different
    parts of the model, making detection more difficult.
    """
    def __init__(self, dataset, num_classes):
        self.dataset = dataset
        self.num_classes = num_classes
        # Create patterns for different sections of the dataset
        dataset_size = len(dataset)
        quarter_size = dataset_size // 4
        
        # Different strategies for different quarters
        self.strategies = [
            lambda x: (x + 1) % num_classes,                  # First quarter: simple shift
            lambda x: (x + num_classes // 2) % num_classes,   # Second quarter: maximum distance
            lambda x: x,                                      # Third quarter: unchanged
            lambda x: num_classes - x - 1                     # Fourth quarter: inversion
        ]
        
        # Map each index to its quarter
        self.quarter_map = np.zeros(dataset_size, dtype=int)
        for i in range(4):
            start = i * quarter_size
            end = (i + 1) * quarter_size if i < 3 else dataset_size
            self.quarter_map[start:end] = i

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, idx):
        data, target = self.dataset[idx]
        # Apply the strategy for this sample's quarter
        quarter = self.quarter_map[idx]
        modified_target = self.strategies[quarter](target)
        return data, modified_target

def load_dataset():
    """Load and prepare the dataset based on configuration."""
    print(f"\nLoading {DATASET} dataset...")
    
    if DATASET == 'MNIST':
        # Ensure data directory exists
        import os
        os.makedirs('data/mnist', exist_ok=True)
        
        # Define transformations
        transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((0.1307,), (0.3081,))
        ])
        
        try:
            # Download and load training data
            train_dataset = datasets.MNIST(
                root='data/mnist',
                train=True,
                download=True,
                transform=transform
            )
            
            # Load test data
            test_dataset = datasets.MNIST(
                root='data/mnist',
                train=False,
                download=True,
                transform=transform
            )
            
            num_classes = 10
            input_channels = 1
        except OSError as e:
            print(f"Error loading MNIST: {e}")
            print("Trying alternative path...")
            
            # Try with absolute path
            import os
            abs_path = os.path.abspath('data/mnist')
            print(f"Using absolute path: {abs_path}")
            
            train_dataset = datasets.MNIST(
                root=abs_path,
                train=True,
                download=True,
                transform=transform
            )
            
            test_dataset = datasets.MNIST(
                root=abs_path,
                train=False,
                download=True,
                transform=transform
            )
            
            num_classes = 10
            input_channels = 1
            
    elif DATASET == 'ALZHEIMER':
        train_dataset, test_dataset, num_classes, input_channels = load_alzheimer_dataset()
    elif DATASET == 'CIFAR10':
        train_dataset, test_dataset, num_classes, input_channels = load_cifar10_dataset()
    else:
        raise ValueError(f"Unknown dataset: {DATASET}")
    
    print(f"Dataset loaded: {len(train_dataset)} training samples, {len(test_dataset)} test samples")
    print(f"Number of classes: {num_classes}, Input channels: {input_channels}")
    
    return train_dataset, test_dataset, num_classes, input_channels

def split_dataset_non_iid(dataset, num_classes):
    """
    Non-IID (Label Skew): Clients have data biased towards certain classes.
    Uses the label skew approach to create non-IID distribution.
    Works with any number of classes and clients.
    """
    # Initialize empty list for each client
    client_datasets = [[] for _ in range(NUM_CLIENTS)]
    
    # Group indices by class
    class_indices = [[] for _ in range(num_classes)]
    for idx, (_, label) in enumerate(dataset):
        class_indices[label].append(idx)
    
    # Shuffle indices within each class
    for c in range(num_classes):
        random.shuffle(class_indices[c])
    
    # Handle both cases: more clients than classes or more classes than clients
    if NUM_CLIENTS <= num_classes:
        # Assign each client a primary class
        classes_per_client = 1
        clients_per_class = max(1, NUM_CLIENTS // num_classes)  # At least 1 client per class
    else:
        # Assign each client multiple primary classes
        classes_per_client = max(1, num_classes // NUM_CLIENTS)  # At least 1 class per client
        clients_per_class = 1
    
    # Create a mapping of classes to their "preferred" clients
    class_to_clients = {}
    for c in range(num_classes):
        # Calculate start and end indices for clients that prefer this class
        start_client = (c * clients_per_class) % NUM_CLIENTS
        end_client = min(start_client + clients_per_class, NUM_CLIENTS)
        
        # Assign these clients to the class
        class_to_clients[c] = list(range(start_client, end_client))
        if not class_to_clients[c]:  # Ensure each class has at least one client
            class_to_clients[c] = [c % NUM_CLIENTS]
    
    # Distribute class samples according to Q parameter
    for c in range(num_classes):
        preferred_clients = class_to_clients[c]
        other_clients = [i for i in range(NUM_CLIENTS) if i not in preferred_clients]
        
        for idx in class_indices[c]:
            if random.random() < Q:
                # Assign to a preferred client for this class
                client_id = random.choice(preferred_clients)
            else:
                # Assign to a non-preferred client
                if other_clients:
                    client_id = random.choice(other_clients)
                else:
                    # If no other clients, assign to a preferred one
                    client_id = random.choice(preferred_clients)
            
            # Add the index to the chosen client's dataset
            client_datasets[client_id].append(idx)
    
    # Convert to PyTorch Subset format
    client_datasets = [torch.utils.data.Subset(dataset, indices) for indices in client_datasets]
    
    # Print statistics about the data distribution
    print("\n=== Data Distribution Statistics ===")
    print(f"Distribution Type: Non-IID (Label Skew) with Q={Q}")
    for i, client_dataset in enumerate(client_datasets):
        labels = [dataset[idx][1] for idx in client_dataset.indices]
        label_counts = {c: labels.count(c) for c in range(num_classes)}
        print(f"Client {i}: {len(client_dataset)} samples, Label distribution: {label_counts}")
    
    return client_datasets

def split_dataset_iid(dataset, num_classes):
    """
    IID: Data is split equally and randomly among clients, ensuring
    each client has a similar class distribution.
    """
    indices = list(range(len(dataset)))
    random.shuffle(indices)
    
    # Calculate the number of samples per client
    samples_per_client = len(indices) // NUM_CLIENTS
    
    # Distribute samples to clients
    client_datasets = []
    for i in range(NUM_CLIENTS):
        start_idx = i * samples_per_client
        end_idx = start_idx + samples_per_client if i < NUM_CLIENTS - 1 else len(indices)
        client_indices = indices[start_idx:end_idx]
        client_datasets.append(torch.utils.data.Subset(dataset, client_indices))
    
    return client_datasets

def split_dataset_dirichlet(dataset, num_classes, alpha=None):
    """
    Non-IID (Dirichlet): Data is split according to a Dirichlet distribution,
    introducing random heterogeneity.
    
    Args:
        dataset: The dataset to split
        num_classes: Number of classes in the dataset
        alpha: Concentration parameter for Dirichlet distribution
              Lower alpha -> more skewed distribution
              If None, uses DIRICHLET_ALPHA from config
    """
    if alpha is None:
        alpha = DIRICHLET_ALPHA
        
    # Get indices for each class
    class_indices = [[] for _ in range(num_classes)]
    for idx, (_, label) in enumerate(dataset):
        class_indices[label].append(idx)
    
    # Shuffle indices within each class
    for c in range(num_classes):
        random.shuffle(class_indices[c])
    
    # Initialize client datasets
    client_datasets = [[] for _ in range(NUM_CLIENTS)]
    
    # Sample from Dirichlet distribution for each client
    proportions = np.random.dirichlet(np.repeat(alpha, NUM_CLIENTS), num_classes)
    
    # Calculate the number of samples from each class for each client
    class_samples_per_client = np.zeros((num_classes, NUM_CLIENTS), dtype=int)
    for c in range(num_classes):
        class_size = len(class_indices[c])
        samples_per_client = np.floor(proportions[c] * class_size).astype(int)
        
        # Adjust to ensure all samples are distributed
        remainder = class_size - samples_per_client.sum()
        if remainder > 0:
            # Add the remaining samples to clients with highest proportions
            indices = np.argsort(proportions[c])[-remainder:]
            for idx in indices:
                samples_per_client[idx] += 1
        
        class_samples_per_client[c] = samples_per_client
    
    # Distribute class samples to clients
    start_idxs = np.zeros(num_classes, dtype=int)
    for client_idx in range(NUM_CLIENTS):
        for class_idx in range(num_classes):
            samples_count = class_samples_per_client[class_idx][client_idx]
            start_idx = start_idxs[class_idx]
            end_idx = start_idx + samples_count
            
            # Get indices for this class and client
            if start_idx < len(class_indices[class_idx]):
                client_class_indices = class_indices[class_idx][start_idx:end_idx]
                client_datasets[client_idx].extend(client_class_indices)
            
            # Update the starting index for next allocation
            start_idxs[class_idx] = end_idx
    
    # Convert to PyTorch Subset format
    client_datasets = [torch.utils.data.Subset(dataset, indices) for indices in client_datasets]
    return client_datasets

def split_dataset(dataset, num_classes, distribution_type=None):
    """
    Split the dataset according to the specified distribution type.
    
    Args:
        dataset: The dataset to split
        num_classes: Number of classes in the dataset
        distribution_type: Type of data distribution ('iid', 'label_skew', 'dirichlet')
                           If None, uses DATA_DISTRIBUTION from config
    
    Returns:
        A list of datasets for each client
    """
    if distribution_type is None:
        distribution_type = DATA_DISTRIBUTION
        
    if distribution_type == 'iid':
        return split_dataset_iid(dataset, num_classes)
    elif distribution_type == 'label_skew':
        return split_dataset_non_iid(dataset, num_classes)
    elif distribution_type == 'dirichlet':
        return split_dataset_dirichlet(dataset, num_classes)
    else:
        raise ValueError(f"Unknown distribution type: {distribution_type}")

def create_root_dataset(full_dataset, num_classes):
    """
    Create a root dataset based on configuration parameters.
    
    Args:
        full_dataset: The full dataset to sample from
        num_classes: Number of classes in the dataset
        
    Returns:
        A subset of the full dataset to be used as the root dataset
    """
    # Determine root dataset size (either fixed or dynamic based on dataset size)
    if ROOT_DATASET_DYNAMIC_SIZE:
        root_size = int(len(full_dataset) * ROOT_DATASET_RATIO)
        print(f"Using dynamic root dataset size: {root_size} samples ({ROOT_DATASET_RATIO*100:.1f}% of total)")
    else:
        root_size = ROOT_DATASET_SIZE
        print(f"Using fixed root dataset size: {root_size} samples")
    
    if BIAS_PROBABILITY == 1.0:
        biased_indices = [i for i, (_, label) in enumerate(full_dataset) if label == BIAS_CLASS]
        root_indices = random.sample(biased_indices, min(root_size, len(biased_indices)))
    else:
        biased_size = int(root_size * BIAS_PROBABILITY)
        unbiased_size = root_size - biased_size
        biased_indices = [i for i, (_, label) in enumerate(full_dataset) if label == BIAS_CLASS]
        unbiased_indices = [i for i, (_, label) in enumerate(full_dataset) if label != BIAS_CLASS]
        root_indices = random.sample(biased_indices, min(biased_size, len(biased_indices))) + \
                       random.sample(unbiased_indices, min(unbiased_size, len(unbiased_indices)))
    
    root_dataset = torch.utils.data.Subset(full_dataset, root_indices)
    return root_dataset 