import argparse
import torch
from torch import nn
from torch import optim
from torchvision import models
from PIL import Image
from torchvision import transforms
import torch.nn.functional as F
from torchvision import datasets, transforms
from torch.utils.data.distributed import DistributedSampler
from torch.utils.data import DataLoader

import os
import numpy as np
import glob
import time

def get_image_paths(root):
  
  count = 0
  index = 0
  image_paths = []
  labels = []

  for dir in os.listdir(root):
    os.chdir(root+'/'+dir)
    count += len(glob.glob('*.JPEG'))
    labels += [index] * len(glob.glob('*.JPEG'))

    for img in glob.glob('*.JPEG'):
      img_path = root+'/'+dir+'/'+img
      image_paths.append(img_path)
      #print(img_path)
  print(count)
  return image_paths

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

def resnet50():
    model = models.resnet50(pretrained=True)
    model = model.to(device)
    root = '/home/vasudev_sridhar007/project/Performance-Comparison-of-TensorFlow-PyTorch-and-their-Distributed-Counterparts/imagenette2/val'
    image_paths = get_image_paths(root)
    natural_img_dataset = datasets.ImageFolder(
                              root = root
                       )
    with open('/home/vasudev_sridhar007/project/Performance-Comparison-of-TensorFlow-PyTorch-and-their-Distributed-Counterparts/imagenet1000_clsidx_to_labels.txt') as f:
        labels = [line.strip() for line in f.readlines()]
    
    dist_sampler = DistributedSampler(natural_img_dataset)
    trainloader = DataLoader(trainset, sampler=dist_sampler)

    preprocess = transforms.Compose([
        #transforms.Resize(259),
        #transforms.CenterCrop(224),
        transforms.ToTensor(),
        #transforms.Normalize(
        #mean=[0.485, 0.456, 0.406],
        #std=[0.229, 0.224, 0.225])
        ])
    t1 = time.time()
    for img in trainloader:
      #'/content/drive/MyDrive/Documents/CIFAR-10-images-master/ILSVRC2012_val_00037861.JPEG'
      ele = Image.open(img).convert('RGB')
      ele = preprocess(ele)
      ele = torch.unsqueeze(ele, 0)
      ele = ele.to(device)
      preds = model(ele)
      _, index = torch.max(preds, 1)
      percentage = torch.nn.functional.softmax(preds, dim=1)[0] * 1000
      print(labels[index[0]], percentage[index[0]].item())

    #_, indices = torch.sort(preds, descending=True)
    #[(labels[idx], percentage[idx].item()) for idx in indices[0][:5]]

    print('Inference Time is: {} seconds'.format(time.time()-t1))

resnet50()

# python3 -m torch.distributed.launch --nproc_per_node=4 --nnodes=2 --node_rank=0 --master_addr="10.182.0.2" --master_port=1234 another_neural_net.py
# python3 -m torch.distributed.launch --nproc_per_node=4 --nnodes=2 --node_rank=1 --master_addr="10.182.0.2" --master_port=1234 another_neural_net.py
