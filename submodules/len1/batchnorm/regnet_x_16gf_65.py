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
        self.batchnorm2d65 = BatchNorm2d(896, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)

    def forward(self, x213):
        x214=self.batchnorm2d65(x213)
        return x214

m = M().eval()
x213 = torch.randn(torch.Size([1, 896, 14, 14]))
start = time.time()
output = m(x213)
end = time.time()
print(end-start)
