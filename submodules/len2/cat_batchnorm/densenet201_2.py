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
        self.batchnorm2d5 = BatchNorm2d(128, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)

    def forward(self, x4, x11, x18):
        x19=torch.cat([x4, x11, x18], 1)
        x20=self.batchnorm2d5(x19)
        return x20

m = M().eval()
x4 = torch.randn(torch.Size([1, 64, 56, 56]))
x11 = torch.randn(torch.Size([1, 32, 56, 56]))
x18 = torch.randn(torch.Size([1, 32, 56, 56]))
start = time.time()
output = m(x4, x11, x18)
end = time.time()
print(end-start)
