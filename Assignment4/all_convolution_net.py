"""
Shubham Arya 1001650536
CSE 4310 Computer Vision
Assignment 4: Image Classification with Deep Learning
All Convolutional Net
Run command- python all_convolution_net.py
"""

import time
import torch
import torch.nn as nn
import torchvision
import numpy as np
import matplotlib.pyplot as plt
from torchvision import transforms
from tqdm.notebook import trange, tqdm
from torch.utils.tensorboard import SummaryWriter
import numpy as np
import torch.nn.functional as F

class Food101(nn.Module):
    def __init__(self, dropout=True, num_classes=101):
        super(Food101, self).__init__()
        self.dropout = dropout
        self.conv1 = nn.Conv2d(3, 6, 5, padding=1)
        self.conv3 = nn.Conv2d(6, 16, 5, padding=1)
        self.class_conv = nn.Conv2d(16, num_classes, 1)


    def forward(self, x):
        if self.dropout:
            x = F.dropout(x, .2)
        conv1_out = F.relu(self.conv1(x))
        conv3_out = F.relu(self.conv3(conv1_out))
        class_out = F.relu(self.class_conv(conv3_out))
        pool_out = class_out.reshape(class_out.size(0), class_out.size(1), -1).mean(-1)
        return pool_out
        
        
class AverageMeter(object):
    """Computes and stores the average and current value"""
    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count


def accuracy(output, target, topk=(1,)):
    """Computes the precision@k for the specified values of k"""
    maxk = max(topk)
    batch_size = target.size(0)

    _, pred = output.topk(maxk, 1, True, True)
    pred = pred.t()
    correct = pred.eq(target.view(1, -1).expand_as(pred))

    res = []
    for k in topk:
        correct_k = correct[:k].view(-1).float().sum(0)
        res.append(correct_k.mul_(100.0 / batch_size))
    return res
    
def numel(m: torch.nn.Module, only_trainable: bool = False):
    parameters = list(m.parameters())
    if only_trainable:
        parameters = [p for p in parameters if p.requires_grad]
    unique = {p.data_ptr(): p for p in parameters}.values()
    return sum(p.numel() for p in unique)
    
# Model Parameters
batch_size = 100
learning_rate = 1e-2
epochs = 50
print_frequency = 50

# device config
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

model = Food101()
print(sum(dict((p.data_ptr(), p.numel()) for p in model.parameters()).values()))

criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.parameters(), learning_rate)

root_path = "./food101data"

normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])

train_transforms = transforms.Compose([
#                                       transforms.RandomRotation(30),
                                       transforms.RandomResizedCrop(224),
#                                       transforms.RandomHorizontalFlip(),
                                       transforms.ToTensor()])
                                       

dataset = torchvision.datasets.Food101(root_path, transform=train_transforms, download=True)

dataset_size = len(dataset)
train_size = int(dataset_size * .95)
val_size = dataset_size - train_size

train_dataset, val_dataset = torch.utils.data.random_split(dataset, [train_size, val_size])

train_dataloader = torch.utils.data.DataLoader(dataset=train_dataset, batch_size=batch_size, shuffle=True)
val_dataloader = torch.utils.data.DataLoader(val_dataset, batch_size=batch_size, shuffle=False, pin_memory=False)
test_dataloader = torch.utils.data.DataLoader(
    torchvision.datasets.Food101(root_path, transform=train_transforms),
    batch_size=batch_size, shuffle=False,pin_memory=False)


    
logger = SummaryWriter("runs/All_Convolution_Net")

n_total_steps = len(train_dataloader)

running_loss = 0.0
running_correct = 0

# training
temp = 0
for epoch in range(epochs):
    losses = AverageMeter()
    top1 = AverageMeter()
    model.train()
    pbar = tqdm(enumerate(train_dataloader), total=len(train_dataloader))
    
    for i, (input, target) in pbar:
    
        output = model(input)
        loss = criterion(output, target)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        output = output.float()
        loss = loss.float()

        prec = accuracy(output.data, target)[0]
        
        running_loss +=  loss.item()
        _, predictions = torch.max(output, 1)
        running_correct += (predictions == target).sum().item()
        
        losses.update(loss.item(), input.shape[0])
        top1.update(prec.item(), input.shape[0])
        
        print("epoch ", epoch," i ",i," loss ",losses.avg)
        
        logger.add_scalar('training loss', losses.avg, epoch*n_total_steps + i)
        logger.add_scalar('training accuracy', top1.avg, epoch*n_total_steps + i)
        temp += 1
        if temp == 50:
            break
    break

print("accuracy of training is ",running_correct / 100)
running_loss = 0.0
running_correct = 0

# validation
temp = 0
for epoch in range(epochs):
    losses = AverageMeter()
    top1 = AverageMeter()

    model.eval()

    pbar = tqdm(enumerate(val_dataloader), total=len(val_dataloader))
    for i, (input, target) in pbar:

        output = model(input)
        loss = criterion(output, target)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        output = output.float()
        loss = loss.float()

        prec = accuracy(output.data, target)[0]
        losses.update(loss.item(), input.shape[0])
        top1.update(prec.item(), input.shape[0])

        running_loss +=  loss.item()
        _, predictions = torch.max(output, 1)
        running_correct += (predictions == target).sum().item()

        print(f'epoch {epoch+1} / {epochs}, step {i+1}/{n_total_steps}, loss=  {loss.item():.4f}')
        logger.add_scalar('validation loss', loss.item(), epoch*n_total_steps + i)
        logger.add_scalar('validation accuracy', top1.avg, epoch*n_total_steps + i)
        temp += 1
        if temp == 100:
            break
    break
  
running_correct = 0

print('Finished Training')
PATH = './all_convolution_net.pth'
torch.save(model.state_dict(), PATH)

print("testing now")
with torch.no_grad():
    n_correct = 0
    n_samples = 0
    n_class_correct = [0 for i in range(101)]
    n_class_samples = [0 for i in range(101)]
    for i, (images, labels) in enumerate(test_dataloader):
        outputs = model(images)
        _, predicted = torch.max(outputs, 1)
        n_samples += labels.size(0)
        n_correct += (predicted == labels).sum().item()
        print("", i," accuracy ",100.0 * n_correct / n_samples)
        
        for j in range(batch_size):
            label = labels[j]
            pred = predicted[j]
            if (label == pred):
                n_class_correct[label] += 1
            n_class_samples[label] += 1

    acc = 100.0 * n_correct / n_samples
    print(f'Accuracy of the network: {acc} %')

    for i in range(101):
        acc = 100.0 * n_class_correct[i] / n_class_samples[i]
        print(f'Accuracy of class {i}: {acc} %')

                          
