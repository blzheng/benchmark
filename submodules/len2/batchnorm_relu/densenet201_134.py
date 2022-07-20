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
        self.batchnorm2d134 = BatchNorm2d(128, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
        self.relu134 = ReLU(inplace=True)

    def forward(self, x473):
        x474=self.batchnorm2d134(x473)
        x475=self.relu134(x474)
        return x475

m = M().eval()
x473 = torch.randn(torch.Size([1, 128, 14, 14]))
start = time.time()
output = m(x473)
end = time.time()
print(end-start)
