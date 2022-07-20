import torch
from torch import tensor
import torch.nn as nn
from torch.nn import *
import torchvision
import torchvision.models as models
from torchvision.ops.stochastic_depth import stochastic_depth
import time
import builtins
import operator

class M(torch.nn.Module):
    def __init__(self):
        super(M, self).__init__()
        self.conv2d161 = Conv2d(56, 1344, kernel_size=(1, 1), stride=(1, 1))
        self.sigmoid25 = Sigmoid()
        self.conv2d162 = Conv2d(1344, 224, kernel_size=(1, 1), stride=(1, 1), bias=False)
        self.batchnorm2d110 = BatchNorm2d(224, eps=0.001, momentum=0.1, affine=True, track_running_stats=True)

    def forward(self, x522, x519):
        x523=self.conv2d161(x522)
        x524=self.sigmoid25(x523)
        x525=operator.mul(x524, x519)
        x526=self.conv2d162(x525)
        x527=self.batchnorm2d110(x526)
        return x527

m = M().eval()
x522 = torch.randn(torch.Size([1, 56, 1, 1]))
x519 = torch.randn(torch.Size([1, 1344, 14, 14]))
start = time.time()
output = m(x522, x519)
end = time.time()
print(end-start)
