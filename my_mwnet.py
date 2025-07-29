import os
import random
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim
import torch.utils.data
import torchvision.transforms as transforms
import torchvision.datasets as datasets
from torch.autograd import Variable
from resnet import ResNet32, VNet
from load_corrupted_data import CIFAR10

torch.manual_seed(1)
random.seed(1)
np.random.seed(1)

use_cuda = torch.cuda.is_available()
device = torch.device("cuda" if use_cuda else "cpu")


def my_dataset():
    normalize = transforms.Normalize(
        mean=[x / 255.0 for x in [125.3, 123.0, 113.9]],
        std=[x / 255.0 for x in [63.0, 62.1, 66.7]]
    )
    
    train_transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Lambda(lambda x: F.pad(x.unsqueeze(0),
                                         (4, 4, 4, 4), mode='reflect').squeeze()),
        transforms.ToPILImage(),
        transforms.RandomCrop(32),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        normalize,
    ])
    
    test_transform = transforms.Compose([
        transforms.ToTensor(),
        normalize
    ])

    train_data_meta = CIFAR10(
        root='../data', train=True, meta=True, num_meta=1000,
        corruption_prob=0.4, corruption_type='unif',
        transform=train_transform, download=True
    )
    train_data = CIFAR10(
        root='../data', train=True, meta=False, num_meta=1000,
        corruption_prob=0.4, corruption_type='unif',
        transform=train_transform, download=True, seed=1
    )
    test_data = CIFAR10(
        root='../data', train=False, transform=test_transform, download=True
    )

    train_loader = torch.utils.data.DataLoader(
        train_data, batch_size=100, shuffle=True,
        num_workers=0, pin_memory=True
    )
    meta_loader = torch.utils.data.DataLoader(
        train_data_meta, batch_size=100, shuffle=True,
        num_workers=0, pin_memory=True
    )
    test_loader = torch.utils.data.DataLoader(
        test_data, batch_size=100, shuffle=False,
        num_workers=0, pin_memory=True
    )

    return train_loader, meta_loader, test_loader


def build_model():
    model = ResNet32(10)
    if use_cuda:
        model.cuda()
        torch.backends.cudnn.benchmark = True
    return model


def accuracy(output, target, topk=(1,)):
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


def adjust_learning_rate(optimizer, epoch):
    current_lr = 1e-1 * ((0.1 ** int(epoch >= 80)) * (0.1 ** int(epoch >= 100)))
    for param_group in optimizer.param_groups:
        param_group['lr'] = current_lr


def test(model, test_loader):
    model.eval()
    correct = 0

    with torch.no_grad():
        for inputs, targets in test_loader:
            inputs, targets = inputs.to(device), targets.to(device)
            outputs = model(inputs)
            _, predicted = outputs.max(1)
            correct += predicted.eq(targets).sum().item()

    return 100. * correct / len(test_loader.dataset)


def train(train_loader, meta_loader, model, vnet, optimizer_model, optimizer_vnet, epoch):
    train_loss = 0.0
    meta_loss = 0.0

    meta_loader_iter = iter(meta_loader)
    for batch_idx, (inputs, targets) in enumerate(train_loader):
        model.train()
        inputs, targets = inputs.to(device), targets.to(device)
        
        meta_model = build_model()
        if use_cuda:
            meta_model.cuda()
        meta_model.load_state_dict(model.state_dict())
        
        outputs = meta_model(inputs)
        cost = F.cross_entropy(outputs, targets, reduce=False)
        cost_v = torch.reshape(cost, (len(cost), 1))
        v_lambda = vnet(cost_v.data)
        l_f_meta = torch.sum(cost_v * v_lambda) / len(cost_v)
        
        meta_model.zero_grad()
        grads = torch.autograd.grad(l_f_meta, (meta_model.params()), create_graph=True)
        meta_lr = 1e-1 * ((0.1 ** int(epoch >= 80)) * (0.1 ** int(epoch >= 100)))
        meta_model.update_params(lr_inner=meta_lr, source_params=grads)
        del grads
        
        try:
            inputs_val, targets_val = next(meta_loader_iter)
        except StopIteration:
            meta_loader_iter = iter(meta_loader)
            inputs_val, targets_val = next(meta_loader_iter)
        inputs_val, targets_val = inputs_val.to(device), targets_val.to(device)
        
        y_g_hat = meta_model(inputs_val)
        l_g_meta = F.cross_entropy(y_g_hat, targets_val)
        
        optimizer_vnet.zero_grad()
        l_g_meta.backward()
        optimizer_vnet.step()
        
        outputs = model(inputs)
        cost_w = F.cross_entropy(outputs, targets, reduce=False)
        cost_v = torch.reshape(cost_w, (len(cost_w), 1))
        
        with torch.no_grad():
            w_new = vnet(cost_v)
        
        loss = torch.sum(cost_v * w_new) / len(cost_v)
        
        optimizer_model.zero_grad()
        loss.backward()
        optimizer_model.step()
        
        train_loss += loss.item()
        meta_loss += l_g_meta.item()


def main():
    train_loader, meta_loader, test_loader = my_dataset()
    
    model = build_model()
    vnet = VNet(1, 100, 1)
    if use_cuda:
        vnet.cuda()
    
    optimizer_model = torch.optim.SGD(
        model.params(), 1e-1,
        momentum=0.9, 
        weight_decay=5e-4,
        nesterov=True
    )
    optimizer_vnet = torch.optim.Adam(
        vnet.params(), 1e-3,
        weight_decay=1e-4
    )
    
    best_acc = 0.0
    for epoch in range(120):
        adjust_learning_rate(optimizer_model, epoch)
        train(train_loader, meta_loader, model, vnet, 
              optimizer_model, optimizer_vnet, epoch)
        test_acc = test(model, test_loader)
        
        if test_acc > best_acc:
            best_acc = test_acc
    
    print(best_acc)


if __name__ == '__main__':
    main()
    