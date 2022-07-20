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
        self.batchnorm2d2 = BatchNorm2d(16, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)

    def forward(self, x13, x26):
        x14=self.batchnorm2d2(x13)
        x27=operator.add(x26, x14)
        return x27

m = M().eval()
x13 = torch.randn(torch.Size([1, 16, 112, 112]))
x26 = torch.randn(torch.Size([1, 16, 112, 112]))
start = time.time()
output = m(x13, x26)
end = time.time()
print(end-start)
