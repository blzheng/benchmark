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
        self.batchnorm2d11 = BatchNorm2d(336, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)

    def forward(self, x40):
        x41=self.batchnorm2d11(x40)
        return x41

m = M().eval()
x40 = torch.randn(torch.Size([1, 336, 56, 56]))
start = time.time()
output = m(x40)
end = time.time()
print(end-start)
