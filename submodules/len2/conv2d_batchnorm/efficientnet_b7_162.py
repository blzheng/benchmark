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
        self.conv2d272 = Conv2d(640, 2560, kernel_size=(1, 1), stride=(1, 1), bias=False)
        self.batchnorm2d162 = BatchNorm2d(2560, eps=0.001, momentum=0.01, affine=True, track_running_stats=True)

    def forward(self, x857):
        x858=self.conv2d272(x857)
        x859=self.batchnorm2d162(x858)
        return x859

m = M().eval()
x857 = torch.randn(torch.Size([1, 640, 7, 7]))
start = time.time()
output = m(x857)
end = time.time()
print(end-start)
