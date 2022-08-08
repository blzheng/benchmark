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
        self.batchnorm2d8 = BatchNorm2d(24, eps=0.001, momentum=0.01, affine=True, track_running_stats=True)

    def forward(self, x26):
        x27=self.batchnorm2d8(x26)
        return x27

m = M().eval()
x26 = torch.randn(torch.Size([1, 24, 56, 56]))
start = time.time()
output = m(x26)
end = time.time()
print(end-start)
