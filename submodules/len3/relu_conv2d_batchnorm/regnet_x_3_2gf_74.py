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
        self.relu74 = ReLU(inplace=True)
        self.conv2d79 = Conv2d(1008, 1008, kernel_size=(1, 1), stride=(1, 1), bias=False)
        self.batchnorm2d79 = BatchNorm2d(1008, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)

    def forward(self, x256):
        x257=self.relu74(x256)
        x258=self.conv2d79(x257)
        x259=self.batchnorm2d79(x258)
        return x259

m = M().eval()
x256 = torch.randn(torch.Size([1, 1008, 7, 7]))
start = time.time()
output = m(x256)
end = time.time()
print(end-start)
