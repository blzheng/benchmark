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
        self.batchnorm2d1 = BatchNorm2d(16, eps=0.001, momentum=0.01, affine=True, track_running_stats=True)
        self.relu0 = ReLU(inplace=True)

    def forward(self, x6):
        x7=self.batchnorm2d1(x6)
        x8=self.relu0(x7)
        return x8

m = M().eval()
x6 = torch.randn(torch.Size([1, 16, 112, 112]))
start = time.time()
output = m(x6)
end = time.time()
print(end-start)
