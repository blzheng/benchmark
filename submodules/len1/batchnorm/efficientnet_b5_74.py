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
        self.batchnorm2d74 = BatchNorm2d(1056, eps=0.001, momentum=0.01, affine=True, track_running_stats=True)

    def forward(self, x388):
        x389=self.batchnorm2d74(x388)
        return x389

m = M().eval()
x388 = torch.randn(torch.Size([1, 1056, 14, 14]))
start = time.time()
output = m(x388)
end = time.time()
print(end-start)
