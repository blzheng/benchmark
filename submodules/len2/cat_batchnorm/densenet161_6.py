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
        self.batchnorm2d13 = BatchNorm2d(384, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)

    def forward(self, x4, x11, x18, x25, x32, x39, x46):
        x47=torch.cat([x4, x11, x18, x25, x32, x39, x46], 1)
        x48=self.batchnorm2d13(x47)
        return x48

m = M().eval()
x4 = torch.randn(torch.Size([1, 96, 56, 56]))
x11 = torch.randn(torch.Size([1, 48, 56, 56]))
x18 = torch.randn(torch.Size([1, 48, 56, 56]))
x25 = torch.randn(torch.Size([1, 48, 56, 56]))
x32 = torch.randn(torch.Size([1, 48, 56, 56]))
x39 = torch.randn(torch.Size([1, 48, 56, 56]))
x46 = torch.randn(torch.Size([1, 48, 56, 56]))
start = time.time()
output = m(x4, x11, x18, x25, x32, x39, x46)
end = time.time()
print(end-start)
