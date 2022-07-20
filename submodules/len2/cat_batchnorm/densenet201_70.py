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
        self.batchnorm2d138 = BatchNorm2d(928, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)

    def forward(self, x481, x488):
        x489=torch.cat([x481, x488], 1)
        x490=self.batchnorm2d138(x489)
        return x490

m = M().eval()
x481 = torch.randn(torch.Size([1, 896, 7, 7]))
x488 = torch.randn(torch.Size([1, 32, 7, 7]))
start = time.time()
output = m(x481, x488)
end = time.time()
print(end-start)
