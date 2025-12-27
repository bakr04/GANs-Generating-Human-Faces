import os
import torch
import torch.nn as nn
import torch.optim as optim
import torch.distributed as dist
import torchvision
import torchvision.transforms as transforms
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data import DataLoader, DistributedSampler
import time

start_time = time.time()

# --- Standard Simple CNN Model ---
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

def setup():
    # 1. Get the local rank (GPU ID 0, 1, etc.) provided by torchrun
    local_rank = int(os.environ["LOCAL_RANK"])
    
    # 2. Set the visible device BEFORE initializing the process group
    # This prevents the "device unknown" warning
    torch.cuda.set_device(local_rank)
    
    # 3. Initialize the process group with the specific device
    dist.init_process_group(
        backend="nccl",
        init_method="env://",
        device_id=torch.device(f"cuda:{local_rank}") 
    )

def cleanup():
    dist.destroy_process_group()

def train():
    setup()
    
    local_rank = int(os.environ["LOCAL_RANK"])
    global_rank = int(os.environ["RANK"])

    # Set Device
    torch.cuda.set_device(local_rank)
    device = torch.device(f"cuda:{local_rank}")

    # --- WANDB SETUP (Master Node Only) ---
    #if global_rank == 0:
        #wandb.init(project="cifar-distributed", name="home-cluster-run")
        #print(f"Training on {world_size} GPUs total.")

    # --- DATASET SETUP (Critical Step) ---
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
    ])

    # 1. Download Handling: Only Rank 0 downloads first
    if global_rank == 0:
        train_set = torchvision.datasets.CIFAR10(root='./data', train=True, download=True, transform=transform)
    
    # 2. Wait for Rank 0 to finish downloading before other nodes proceed
    dist.barrier()

    # 3. Everyone loads the dataset (now that it's guaranteed to exist)
    if global_rank != 0:
        train_set = torchvision.datasets.CIFAR10(root='./data', train=True, download=False, transform=transform)

    # 4. The Sampler splits the data among GPUs
    sampler = DistributedSampler(train_set)
    
    # 5. DataLoader needs pin_memory=True for speed
    train_loader = DataLoader(train_set, batch_size=128, shuffle=False, 
                              num_workers=2, sampler=sampler, pin_memory=True)

    # --- MODEL SETUP ---
    model = SimpleCNN().to(device)
    model = DDP(model, device_ids=[local_rank])

    criterion = nn.CrossEntropyLoss()
    optimizer = optim.SGD(model.parameters(), lr=0.01, momentum=0.9)

    # --- TRAINING LOOP ---
    for epoch in range(10):
        # REQUIRED: Set epoch so shuffling changes each time
        sampler.set_epoch(epoch)
        
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
        
        # --- LOGGING (Master Only) ---
        if global_rank == 0:
            print(f"Epoch [{epoch+1}/10] Loss: {avg_loss:.4f}")
            #wandb.log({"epoch": epoch, "loss": avg_loss})
    
    # Save checkpoint only on Master
    if global_rank == 0:
        torch.save(model.module.state_dict(), "cifar_model.pth")
        print("Model saved.")

    cleanup()

if __name__ == "__main__":
    train()
    print(time.time() - start_time)


"""
For Linux Only!


export NCCL_SOCKET_IFNAME=wlp2s0
export OMP_NUM_THREADS=16

torchrun \
    --nproc_per_node=1 \
    --nnodes=2 \
    --node_rank=0 \
    --master_addr=192.168.1.223 \
    --master_port=29500 \
    train_cifar_ddp.py


    



export NCCL_SOCKET_IFNAME=wlp2s0
export OMP_NUM_THREADS=16

torchrun \
    --nproc_per_node=1 \
    --nnodes=2 \
    --node_rank=1 \
    --master_addr=192.168.1.223 \
    --master_port=29500 \
    train_cifar_ddp.py
"""
