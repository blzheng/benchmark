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
        self.conv2d73 = Conv2d(2520, 2520, kernel_size=(1, 1), stride=(1, 1), bias=False)
        self.batchnorm2d73 = BatchNorm2d(2520, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
        self.relu69 = ReLU(inplace=True)

    def forward(self, x237, x231):
        x238=self.conv2d73(x237)
        x239=self.batchnorm2d73(x238)
        x240=operator.add(x231, x239)
        x241=self.relu69(x240)
        return x241

m = M().eval()
x237 = torch.randn(torch.Size([1, 2520, 7, 7]))
x231 = torch.randn(torch.Size([1, 2520, 7, 7]))
start = time.time()
output = m(x237, x231)
end = time.time()
print(end-start)