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
        self.relu67 = ReLU(inplace=True)
        self.conv2d71 = Conv2d(256, 256, kernel_size=(3, 3), stride=(1, 1), padding=(2, 2), dilation=(2, 2), bias=False)
        self.batchnorm2d71 = BatchNorm2d(256, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
        self.conv2d72 = Conv2d(256, 1024, kernel_size=(1, 1), stride=(1, 1), bias=False)
        self.batchnorm2d72 = BatchNorm2d(1024, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)

    def forward(self, x234):
        x235=self.relu67(x234)
        x236=self.conv2d71(x235)
        x237=self.batchnorm2d71(x236)
        x238=self.relu67(x237)
        x239=self.conv2d72(x238)
        x240=self.batchnorm2d72(x239)
        return x240

m = M().eval()
x234 = torch.randn(torch.Size([1, 256, 28, 28]))
start = time.time()
output = m(x234)
end = time.time()
print(end-start)
