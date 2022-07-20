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
        self.batchnorm2d36 = BatchNorm2d(480, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
        self.relu24 = ReLU(inplace=True)

    def forward(self, x118):
        x119=self.batchnorm2d36(x118)
        x120=self.relu24(x119)
        return x120

m = M().eval()
x118 = torch.randn(torch.Size([1, 480, 14, 14]))
start = time.time()
output = m(x118)
end = time.time()
print(end-start)
