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
        self.batchnorm2d0 = BatchNorm2d(40, eps=1e-05, momentum=0.00029999999999996696, affine=True, track_running_stats=True)
        self.relu0 = ReLU(inplace=True)

    def forward(self, x1):
        x2=self.batchnorm2d0(x1)
        x3=self.relu0(x2)
        return x3

m = M().eval()
x1 = torch.randn(torch.Size([1, 40, 112, 112]))
start = time.time()
output = m(x1)
end = time.time()
print(end-start)
