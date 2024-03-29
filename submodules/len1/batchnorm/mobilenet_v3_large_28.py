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
        self.batchnorm2d28 = BatchNorm2d(184, eps=0.001, momentum=0.01, affine=True, track_running_stats=True)

    def forward(self, x100):
        x101=self.batchnorm2d28(x100)
        return x101

m = M().eval()
x100 = torch.randn(torch.Size([1, 184, 14, 14]))
start = time.time()
output = m(x100)
end = time.time()
print(end-start)
