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
        self.batchnorm2d31 = BatchNorm2d(160, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
        self.relu28 = ReLU(inplace=True)

    def forward(self, x100):
        x101=self.batchnorm2d31(x100)
        x102=self.relu28(x101)
        return x102

m = M().eval()
x100 = torch.randn(torch.Size([1, 160, 14, 14]))
start = time.time()
output = m(x100)
end = time.time()
print(end-start)
