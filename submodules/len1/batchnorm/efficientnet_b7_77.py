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
        self.batchnorm2d77 = BatchNorm2d(160, eps=0.001, momentum=0.01, affine=True, track_running_stats=True)

    def forward(self, x412):
        x413=self.batchnorm2d77(x412)
        return x413

m = M().eval()
x412 = torch.randn(torch.Size([1, 160, 14, 14]))
start = time.time()
output = m(x412)
end = time.time()
print(end-start)
