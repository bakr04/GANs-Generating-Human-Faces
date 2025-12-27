import torch
import torch.nn as nn
import torch.optim as optim
import torchvision
import torchvision.transforms as transforms
from torch.utils.data import DataLoader
import time

start_time = time.time()

# --- Standard Simple CNN Model (Unchanged) ---
class SimpleCNN(nn.Module):
    def __init__(self):
        super(SimpleCNN, self).__init__()
        self.conv1 = nn.Conv2d(3, 32, 3)
        self.pool = nn.MaxPool2d(2, 2)
        self.conv2 = nn.Conv2d(32, 64, 3)
        self.fc1 = nn.Linear(64 * 6 * 6, 64)
        self.fc2 = nn.Linear(64, 10)

    def forward(self, x):
        x = self.pool(torch.relu(self.conv1(x)))
        x = self.pool(torch.relu(self.conv2(x)))
        x = x.view(-1, 64 * 6 * 6)
        x = torch.relu(self.fc1(x))
        x = self.fc2(x)
        return x

def train():
    # 1. Device Selection (Automatic)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Training on: {device}")

    # --- DATASET SETUP ---
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
    ])

    # 2. Standard Dataset & Loader
    # No barrier needed, download=True handles locking internally usually, 
    # but for single process it's safe.
    train_set = torchvision.datasets.CIFAR10(root='./data', train=True, 
                                           download=True, transform=transform)
    
    # shuffle=True is crucial here (replaced DistributedSampler)
    train_loader = DataLoader(train_set, batch_size=128, shuffle=True, 
                              num_workers=2, pin_memory=True)

    # --- MODEL SETUP ---
    model = SimpleCNN().to(device)
    # DDP wrapper removed

    criterion = nn.CrossEntropyLoss()
    optimizer = optim.SGD(model.parameters(), lr=0.01, momentum=0.9)

    # --- TRAINING LOOP ---
    for epoch in range(10):
        # sampler.set_epoch(epoch) removed
        
        running_loss = 0.0
        model.train()
        
        for i, (images, labels) in enumerate(train_loader):
            images, labels = images.to(device), labels.to(device)

            optimizer.zero_grad()
            outputs = model(images)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            running_loss += loss.item()

        # Calculate average loss across the epoch
        avg_loss = running_loss / len(train_loader)
        
        print(f"Epoch [{epoch+1}/10] Loss: {avg_loss:.4f}")
    
    # Save checkpoint
    torch.save(model.state_dict(), "cifar_model.pth")
    print("Model saved.")

if __name__ == "__main__":
    train()
    print(f"Total time: {time.time() - start_time:.2f} seconds")