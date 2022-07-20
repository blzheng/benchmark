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
        self.batchnorm2d60 = BatchNorm2d(192, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
        self.relu60 = ReLU(inplace=True)

    def forward(self, x214):
        x215=self.batchnorm2d60(x214)
        x216=self.relu60(x215)
        return x216

m = M().eval()
x214 = torch.randn(torch.Size([1, 192, 14, 14]))
start = time.time()
output = m(x214)
end = time.time()
print(end-start)