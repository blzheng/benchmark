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
        self.sigmoid13 = Sigmoid()
        self.conv2d74 = Conv2d(440, 440, kernel_size=(1, 1), stride=(1, 1), bias=False)
        self.batchnorm2d46 = BatchNorm2d(440, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
        self.relu56 = ReLU(inplace=True)
        self.conv2d75 = Conv2d(440, 440, kernel_size=(1, 1), stride=(1, 1), bias=False)
        self.batchnorm2d47 = BatchNorm2d(440, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)

    def forward(self, x229, x225, x219):
        x230=self.sigmoid13(x229)
        x231=operator.mul(x230, x225)
        x232=self.conv2d74(x231)
        x233=self.batchnorm2d46(x232)
        x234=operator.add(x219, x233)
        x235=self.relu56(x234)
        x236=self.conv2d75(x235)
        x237=self.batchnorm2d47(x236)
        return x237

m = M().eval()
x229 = torch.randn(torch.Size([1, 440, 1, 1]))
x225 = torch.randn(torch.Size([1, 440, 7, 7]))
x219 = torch.randn(torch.Size([1, 440, 7, 7]))
start = time.time()
output = m(x229, x225, x219)
end = time.time()
print(end-start)
