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
        self.batchnorm2d36 = BatchNorm2d(1024, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
        self.relu31 = ReLU(inplace=True)

    def forward(self, x117, x110):
        x118=self.batchnorm2d36(x117)
        x119=operator.add(x118, x110)
        x120=self.relu31(x119)
        return x120

m = M().eval()
x117 = torch.randn(torch.Size([1, 1024, 14, 14]))
x110 = torch.randn(torch.Size([1, 1024, 14, 14]))
start = time.time()
output = m(x117, x110)
end = time.time()
print(end-start)
