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
    local_rank = int(os.environ["LOCAL_RANK"])
    
    torch.cuda.set_device(local_rank)
    
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

    torch.cuda.set_device(local_rank)
    device = torch.device(f"cuda:{local_rank}")


    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
    ])

    if global_rank == 0:
        train_set = torchvision.datasets.CIFAR10(root='./data', train=True, download=True, transform=transform)
    
    dist.barrier()

    if global_rank != 0:
        train_set = torchvision.datasets.CIFAR10(root='./data', train=True, download=False, transform=transform)

    sampler = DistributedSampler(train_set)
    
    train_loader = DataLoader(train_set, batch_size=128, shuffle=False, 
                              num_workers=2, sampler=sampler, pin_memory=True)

    model = SimpleCNN().to(device)
    model = DDP(model, device_ids=[local_rank])

    criterion = nn.CrossEntropyLoss()
    optimizer = optim.SGD(model.parameters(), lr=0.01, momentum=0.9)

    for epoch in range(10):
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

        avg_loss = running_loss / len(train_loader)
        
        if global_rank == 0:
            print(f"Epoch [{epoch+1}/10] Loss: {avg_loss:.4f}")
    
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
