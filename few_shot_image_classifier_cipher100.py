!set PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, SubsetRandomSampler
from torchvision.datasets import CIFAR100
from torchvision.transforms import ToTensor, RandomHorizontalFlip, RandomVerticalFlip, RandomRotation
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
import numpy as np
import torch.optim.lr_scheduler as lr_scheduler


import torchvision.models as models


# Initialize model, loss function, optimizer
model = models.resnet101(pretrained=False)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model.to(device)
loss_fn = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=0.0001)

# Define the scheduler
scheduler = lr_scheduler.StepLR(optimizer, step_size=30, gamma=0.1)


def create_episode_dataset(dataset, n_way, k_shot, query_set_size):
    classes = np.random.choice(len(dataset.classes), n_way, replace=False)
    support_set_indices = []
    query_set_indices = []
    for class_idx in classes:
        class_indices = np.where(np.array(dataset.targets) == class_idx)[0]
        if len(class_indices) == 0:
            continue  # Skip classes with no samples
        support_set_indices.append(np.random.choice(class_indices, k_shot, replace=False)[0])  # Select 1 image for support set  ####  changed 1 with k_shot  ###########
        # Adjust query_set_size if it exceeds the number of samples available
        actual_query_set_size = min(query_set_size, len(class_indices))
        query_set_indices.extend(np.random.choice(class_indices, actual_query_set_size, replace=False))  # Select multiple images for query set
    return support_set_indices, query_set_indices



# Function to train the model on an episode (support set)
def train_episode(model, dataset, support_set_indices, optimizer, loss_fn, device):
    model.train()
    support_loader = DataLoader(dataset, batch_size=len(support_set_indices), sampler=SubsetRandomSampler(support_set_indices))
    for data, targets in support_loader:
        data, targets = data.to(device), targets.to(device)
        optimizer.zero_grad()
        outputs = model(data)
        loss = loss_fn(outputs, targets)
        loss.backward()
        optimizer.step()


# Function to evaluate the model on an episode (query set)
def evaluate_episode(model, dataset, query_set_indices, device):
    model.eval()
    query_loader = DataLoader(dataset, batch_size=len(query_set_indices), sampler=SubsetRandomSampler(query_set_indices))
    for data, targets in query_loader:
        data, targets = data.to(device), targets.to(device)
        outputs = model(data)
        _, predicted = torch.max(outputs.data, 1)
        return predicted.cpu().numpy(), targets.cpu().numpy()


# Define custom dataset class with data augmentation
class AugmentedCIFAR100(CIFAR100):
    def __init__(self, root, train=True, transform=None, target_transform=None, download=False):
        super(AugmentedCIFAR100, self).__init__(root, train=train, transform=transform, target_transform=target_transform, download=download)
        self.augmented_transform = nn.Sequential(
            RandomHorizontalFlip(),
            RandomVerticalFlip(),
            RandomRotation(30)
        )

    def __getitem__(self, index):
        img, target = super(AugmentedCIFAR100, self).__getitem__(index)
        if self.train:  # Apply augmentation only during training
            img = self.augmented_transform(img)
        return img, target


num_episodes = 5
n_way = 100
k_shot = 1
query_set_size = 50
num_epochs = 200

# Load augmented CIFAR-100 dataset
train_dataset = AugmentedCIFAR100("CIFAR100_dataset", transform=ToTensor(), train=True, download=True)
test_dataset = CIFAR100("CIFAR100_dataset", transform=ToTensor(), train=False)

# # Modify training loop to handle episodes
# for epoch in range(num_epochs):
#     for episode in range(num_episodes):
#         support_set_indices, query_set_indices = create_episode_dataset(train_dataset, n_way, k_shot, query_set_size)
#         train_episode(model, train_dataset, support_set_indices, optimizer, loss_fn, device)
#         predictions, targets = evaluate_episode(model, train_dataset, query_set_indices, device)
#         accuracy = accuracy_score(targets, predictions)
#         precision = precision_score(targets, predictions, average='macro', zero_division=0)
#         recall = recall_score(targets, predictions, average='macro', zero_division=0)
#         f1 = f1_score(targets, predictions, average='macro')
#         print(f"Epoch [{epoch+1}/{num_epochs}], Episode [{episode+1}/{num_episodes}], Accuracy: {round(accuracy,4)}, Precision: {round(precision,4)}, Recall: {round(recall,4)}, F1-score: {round(f1,4)}")
    
#     # Step the scheduler at the end of each epoch
#     scheduler.step()



# Define a smaller batch size
batch_size = 4  # Adjust this value based on your GPU memory capacity

# Modify training loop to handle episodes
for epoch in range(num_epochs):
    for episode in range(num_episodes):
        support_set_indices, query_set_indices = create_episode_dataset(train_dataset, n_way, k_shot, query_set_size)
        # Create data loaders with the reduced batch size
        support_loader = DataLoader(train_dataset, batch_size=batch_size, sampler=SubsetRandomSampler(support_set_indices))
        query_loader = DataLoader(train_dataset, batch_size=batch_size, sampler=SubsetRandomSampler(query_set_indices))
        
        # Train the model with the reduced batch size
        for data, targets in support_loader:
            data, targets = data.to(device), targets.to(device)
            optimizer.zero_grad()
            outputs = model(data)
            loss = loss_fn(outputs, targets)
            loss.backward()
            optimizer.step()

        # Evaluate the model with the reduced batch size
        predictions, targets = [], []
        for data, target in query_loader:
            data, target = data.to(device), target.to(device)
            output = model(data)
            _, predicted = torch.max(output, 1)
            predictions.extend(predicted.cpu().numpy())
            targets.extend(target.cpu().numpy())

        accuracy = accuracy_score(targets, predictions)
        precision = precision_score(targets, predictions, average='macro', zero_division=0)
        recall = recall_score(targets, predictions, average='macro', zero_division=0)
        f1 = f1_score(targets, predictions, average='macro')
        print(f"Epoch [{epoch+1}/{num_epochs}], Episode [{episode+1}/{num_episodes}], Accuracy: {round(accuracy,4)}, Precision: {round(precision,4)}, Recall: {round(recall,4)}, F1-score: {round(f1,4)}")
    
    # Step the scheduler at the end of each epoch
    scheduler.step()




# Modify evaluation loop to handle episodes
targets_list = []
predictions_list = []

for _ in range(num_episodes):
    support_set_indices, query_set_indices = create_episode_dataset(test_dataset, n_way, k_shot, query_set_size)
    episode_predictions, episode_targets = evaluate_episode(model, test_dataset, query_set_indices, device)
    targets_list.append(episode_targets)
    predictions_list.append(episode_predictions)

# Calculate evaluation metrics
targets = np.concatenate(targets_list)
predictions = np.concatenate(predictions_list)
accuracy = accuracy_score(targets, predictions)
precision = precision_score(targets, predictions, average='macro', zero_division=0)
recall = recall_score(targets, predictions, average='macro', zero_division=0)
f1 = f1_score(targets, predictions, average='macro')

print(f"Test Accuracy: {accuracy}, Precision: {precision}, Recall: {recall}, F1-score: {f1}")
