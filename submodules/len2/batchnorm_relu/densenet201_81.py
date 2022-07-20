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
        self.batchnorm2d81 = BatchNorm2d(928, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
        self.relu81 = ReLU(inplace=True)

    def forward(self, x288):
        x289=self.batchnorm2d81(x288)
        x290=self.relu81(x289)
        return x290

m = M().eval()
x288 = torch.randn(torch.Size([1, 928, 14, 14]))
start = time.time()
output = m(x288)
end = time.time()
print(end-start)
