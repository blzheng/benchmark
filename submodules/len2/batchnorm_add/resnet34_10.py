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
        self.batchnorm2d20 = BatchNorm2d(256, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)

    def forward(self, x68, x64):
        x69=self.batchnorm2d20(x68)
        x70=operator.add(x69, x64)
        return x70

m = M().eval()
x68 = torch.randn(torch.Size([1, 256, 14, 14]))
x64 = torch.randn(torch.Size([1, 256, 14, 14]))
start = time.time()
output = m(x68, x64)
end = time.time()
print(end-start)
