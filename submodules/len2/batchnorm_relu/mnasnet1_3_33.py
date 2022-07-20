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
        self.batchnorm2d49 = BatchNorm2d(1488, eps=1e-05, momentum=0.00029999999999996696, affine=True, track_running_stats=True)
        self.relu33 = ReLU(inplace=True)

    def forward(self, x142):
        x143=self.batchnorm2d49(x142)
        x144=self.relu33(x143)
        return x144

m = M().eval()
x142 = torch.randn(torch.Size([1, 1488, 7, 7]))
start = time.time()
output = m(x142)
end = time.time()
print(end-start)