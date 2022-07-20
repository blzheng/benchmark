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
        self.batchnorm2d73 = BatchNorm2d(1920, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
        self.relu69 = ReLU(inplace=True)

    def forward(self, x238, x231):
        x239=self.batchnorm2d73(x238)
        x240=operator.add(x231, x239)
        x241=self.relu69(x240)
        return x241

m = M().eval()
x238 = torch.randn(torch.Size([1, 1920, 7, 7]))
x231 = torch.randn(torch.Size([1, 1920, 7, 7]))
start = time.time()
output = m(x238, x231)
end = time.time()
print(end-start)
