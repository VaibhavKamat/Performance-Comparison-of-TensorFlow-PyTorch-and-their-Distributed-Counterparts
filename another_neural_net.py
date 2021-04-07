import argparse
import torch
from torch import nn
from torch import optim
import torch.nn.functional as F
from torchvision import datasets, transforms
from torch.utils.data.distributed import DistributedSampler
from torch.utils.data import DataLoader

# prase the local_rank argument from command line for the current process
parser = argparse.ArgumentParser()
parser.add_argument("--local_rank", default=0, type=int)
args = parser.parse_args()

# setup the distributed backend for managing the distributed training
torch.distributed.init_process_group('gloo')
# Define a transform to normalize the data
transform = transforms.Compose([transforms.ToTensor(),
                                transforms.Normalize((0.5,), (0.5,)),
                              ])
# Download and load the training data
trainset = datasets.MNIST('~/.pytorch/MNIST_data/', download=True, train=True, transform=transform)
# trainloader = torch.utils.data.DataLoader(trainset, batch_size=64, shuffle=True)

# Setup the distributed sampler to split the dataset to each GPU.
dist_sampler = DistributedSampler(trainset)
trainloader = DataLoader(trainset, sampler=dist_sampler)


if torch.cuda.is_available():
    print("Cuda Device Available")
    device_ids = list(range(torch.cuda.device_count()))
    print(device_ids)
    device = torch.device("cuda", args.local_rank)
    print("Name of the Cuda Device: ", torch.cuda.get_device_name())
    print("GPU Computational Capablity: ", torch.cuda.get_device_capability())
else:
    device = torch.device("cpu", args.local_rank)
    print('No GPU. switching to CPU')

model = nn.Sequential(nn.Linear(784, 128),
                      nn.ReLU(),
                      nn.Linear(128, 64),
                      nn.ReLU(),
                      nn.Linear(64, 10),
                      nn.LogSoftmax(dim=1))

model = model.to(device)
# Define the loss
criterion = nn.NLLLoss()
# Optimizers require the parameters to optimize and a learning rate
optimizer = optim.SGD(model.parameters(), lr=0.003)
epochs = 5
for e in range(epochs):
    running_loss = 0
    for images, labels in trainloader:
        # Flatten MNIST images into a 784 long vector
        images = images.view(images.shape[0], -1)
        images = images.to(device)
        labels = labels.to(device)
        # Training pass
        optimizer.zero_grad()

        output = model(images)
        loss = criterion(output, labels)
        loss.backward()
        optimizer.step()

        running_loss += loss.item()
    else:
        print(f"Training loss: {running_loss / len(trainloader)}")

# python3 -m torch.distributed.launch --nproc_per_node=4 --nnodes=2 --node_rank=0 --master_addr="10.182.0.2" --master_port=1234 another_neural_net.py
# python3 -m torch.distributed.launch --nproc_per_node=4 --nnodes=2 --node_rank=1 --master_addr="10.182.0.2" --master_port=1234 another_neural_net.py
