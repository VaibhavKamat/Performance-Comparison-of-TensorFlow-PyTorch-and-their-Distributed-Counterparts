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

def load_split_train_test(datadir, valid_size = .2):
    train_transforms = transforms.Compose([transforms.Resize((224,224)),
                                       transforms.ToTensor(),
                                       ])
    test_transforms = transforms.Compose([transforms.Resize((224,224)),
                                      transforms.ToTensor(),
                                      ])
    train_data = datasets.ImageFolder(datadir,
                    transform=train_transforms)
    test_data = datasets.ImageFolder(datadir,
                    transform=test_transforms)
    num_train = len(train_data)
    indices = list(range(num_train))
    split = int(np.floor(valid_size * num_train))
    np.random.shuffle(indices)

    train_idx, test_idx = indices[split:], indices[:split]
    dist_sampler_train = DistributedSampler(train_idx)
    dist_sampler_test = DistributedSampler(test_idx)

    trainloader = torch.utils.data.DataLoader(train_data,
                   sampler=dist_sampler_train, batch_size=64)
    testloader = torch.utils.data.DataLoader(test_data,
                   sampler=dist_sampler_test, batch_size=64)
    return trainloader, testloader

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

def resnet50(device):
    model = models.resnet50(pretrained=True)

    data_dir = '/home/vasudev_sridhar007/project/Performance-Comparison-of-TensorFlow-PyTorch-and-their-Distributed-Counterparts/imagenette2/train'
    root = '/home/vasudev_sridhar007/project/Performance-Comparison-of-TensorFlow-PyTorch-and-their-Distributed-Counterparts/imagenette2/val'

    trainloader, testloader = load_split_train_test(data_dir, .2)

    preprocess = transforms.Compose([
        #transforms.Resize(259),
        #transforms.CenterCrop(224),
        transforms.ToTensor(),
        #transforms.Normalize(
        #mean=[0.485, 0.456, 0.406],
        #std=[0.229, 0.224, 0.225])
        ])
    for param in model.parameters():
        param.requires_grad = False

    model.fc = nn.Sequential(nn.Linear(2048, 512),
                             nn.ReLU(),
                             nn.Dropout(0.2),
                             nn.Linear(512, 10),
                             nn.LogSoftmax(dim=1))
    criterion = nn.NLLLoss()
    optimizer = optim.Adam(model.fc.parameters(), lr=0.003)
    model = model.to(device)

    t1 = time.time()
    epochs = 1
    steps = 0
    running_loss = 0
    print_every = 10
    train_losses, test_losses = [], []
    for epoch in range(epochs):
        print("Hi")
        for inputs, labels in trainloader:
            steps += 1
            print(labels)
            print(steps)
            inputs, labels = inputs.to(device), labels.to(device)
            optimizer.zero_grad()
            logps = model.forward(inputs)
            loss = criterion(logps, labels)
            loss.backward()
            optimizer.step()
            running_loss += loss.item()
        print("trainloader done")
        # if steps % print_every == 0:
        test_loss = 0
        accuracy = 0
        model.eval()
        with torch.no_grad():

            for inputs, labels in testloader:
                inputs, labels = inputs.to(device), labels.to(device)
                logps = model.forward(inputs)
                batch_loss = criterion(logps, labels)
                test_loss += batch_loss.item()

                ps = torch.exp(logps)
                top_p, top_class = ps.topk(1, dim=1)
                equals = top_class == labels.view(*top_class.shape)
                accuracy += torch.mean(equals.type(torch.FloatTensor)).item()
        train_losses.append(running_loss / len(trainloader))
        test_losses.append(test_loss / len(testloader))
        print(f"Epoch {epoch + 1}/{epochs}.. "
              f"Train loss: {running_loss / print_every:.3f}.. "
              f"Test loss: {test_loss / len(testloader):.3f}.. "
              f"Test accuracy: {accuracy / len(testloader):.3f}")
        running_loss = 0
        model.train()
    #print("Saving Model")
    #torch.save(model,
    #           '/home/vasudev_sridhar007/project/Performance-Comparison-of-TensorFlow-PyTorch-and-their-Distributed-Counterparts/imagenette2/aerialmodel.pth')  #########NEED TO CHANGE THIS PATH ACC TO GCP DIRS

    print("Training time per epoch is {} seconds".format(time.time() - t1))

    ####

    from torch.autograd import Variable
    test_transforms = transforms.Compose([transforms.Resize((224, 224)),
                                          transforms.ToTensor()
                                          ])

    # device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    #model = torch.load(
    #    '/home/vasudev_sridhar007/project/Performance-Comparison-of-TensorFlow-PyTorch-and-their-Distributed-Counterparts/imagenette2/aerialmodel.pth')  #########NEED TO CHANGE THIS PATH ACC TO GCP DIRS
    #model.eval()
    model = model.to(device)

    def predict_image(image):
        image_tensor = test_transforms(image).float()
        image_tensor = image_tensor.unsqueeze_(0)
        input = Variable(image_tensor)
        input = input.to(device)
        output = model(input)
        index = output.data.cpu().numpy().argmax()
        return index

    def get_random_images(num):
        data = datasets.ImageFolder(data_dir, transform=test_transforms)
        classes = data.classes
        indices = list(range(len(data)))
        np.random.shuffle(indices)
        idx = indices[:num]

        sampler = DistributedSampler(idx)
        loader = torch.utils.data.DataLoader(data,
                                             sampler=sampler, batch_size=num)
        dataiter = iter(loader)
        images, labels = dataiter.next()
        return images, labels

    t1 = time.time()
    to_pil = transforms.ToPILImage()
    images, labels = get_random_images(1000)
    # fig=plt.figure(figsize=(10,10))
    for ii in range(len(images)):
        print(ii + 1)
        image = to_pil(images[ii])
        index = predict_image(image)
        # sub = fig.add_subplot(1, len(images), ii+1)
        res = int(labels[ii]) == index
        # sub.set_title(str(trainloader.dataset.classes[index]) + ":" + str(res))
        # plt.axis('off')
        ##plt.imshow(image)
    # plt.show()
    print("Inference time is {} seconds".format(time.time() - t1))

resnet50(device)

# python3 -m torch.distributed.launch --nproc_per_node=4 --nnodes=2 --node_rank=0 --master_addr="10.182.0.2" --master_port=1234 another_neural_net.py
# python3 -m torch.distributed.launch --nproc_per_node=4 --nnodes=2 --node_rank=1 --master_addr="10.182.0.2" --master_port=1234 another_neural_net.py
