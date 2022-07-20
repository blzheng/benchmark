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
        self.batchnorm2d73 = BatchNorm2d(1056, eps=0.001, momentum=0.01, affine=True, track_running_stats=True)

    def forward(self, x385):
        x386=self.batchnorm2d73(x385)
        return x386

m = M().eval()
x385 = torch.randn(torch.Size([1, 1056, 14, 14]))
start = time.time()
output = m(x385)
end = time.time()
print(end-start)