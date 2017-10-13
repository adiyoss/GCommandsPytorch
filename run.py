from __future__ import print_function
import argparse
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torchvision import datasets, transforms
from torch.autograd import Variable
from gcommand_loader import GCommandLoader
from model import Net
from train import train, test

# Training settings
parser = argparse.ArgumentParser(description='PyTorch MNIST Example')
parser.add_argument('--train_path', default='gcommands/train', help='path to the train data folder')
parser.add_argument('--test_path', default='gcommands/test', help='path to the test data folder')
parser.add_argument('--valid_path', default='gcommands/valid', help='path to the valid data folder')
parser.add_argument('--batch_size', type=int, default=100, metavar='N', help='training and valid batch size')
parser.add_argument('--test_batch_size', type=int, default=100, metavar='N', help='batch size for testing')
parser.add_argument('--epochs', type=int, default=10, metavar='N', help='number of epochs to train')
parser.add_argument('--lr', type=float, default=0.001, metavar='LR', help='learning rate')
parser.add_argument('--momentum', type=float, default=0.9, metavar='M', help='SGD momentum, for SGD only')
parser.add_argument('--optimizer', default='adam', help='optimization method: sgd | adam')
parser.add_argument('--cuda', default=False, help='enable CUDA')
parser.add_argument('--seed', type=int, default=1234, metavar='S', help='random seed')
parser.add_argument('--log-interval', type=int, default=10, metavar='N', help='how many batches to wait before logging training status')
args = parser.parse_args()

args.cuda = args.cuda and torch.cuda.is_available()
torch.manual_seed(args.seed)
if args.cuda:
    torch.cuda.manual_seed(args.seed)

# loading data
train_dataset = GCommandLoader(args.train_path)
train_loader = torch.utils.data.DataLoader(
        train_dataset, batch_size=args.batch_size, shuffle=True,
        num_workers=20, pin_memory=True, sampler=None)

valid_dataset = GCommandLoader(args.valid_path)
valid_loader = torch.utils.data.DataLoader(
        valid_dataset, batch_size=args.batch_size, shuffle=None,
        num_workers=20, pin_memory=True, sampler=None)

test_dataset = GCommandLoader(args.test_path)
test_loader = torch.utils.data.DataLoader(
        test_dataset, batch_size=args.test_batch_size, shuffle=None,
        num_workers=20, pin_memory=True, sampler=None)

# build model
model = Net()
if args.cuda:
    print('Using CUDA')
    model.cuda()

# define optimizer
if args.optimizer.lower() == 'adam':
    optimizer = optim.Adam(model.parameters(), lr=args.lr)
elif args.optimizer.lower() == 'sgd':
    optimizer = optim.SGD(model.parameters(), lr=args.lr, momentum=args.momentum)

for epoch in range(1, args.epochs + 1):
    train(train_loader, model, optimizer, epoch, args.cuda, args.log_interval)
    test(test_loader, args.cuda)
