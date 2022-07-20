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
        self.batchnorm2d22 = BatchNorm2d(192, eps=1e-05, momentum=0.00029999999999996696, affine=True, track_running_stats=True)
        self.relu15 = ReLU(inplace=True)

    def forward(self, x64):
        x65=self.batchnorm2d22(x64)
        x66=self.relu15(x65)
        return x66

m = M().eval()
x64 = torch.randn(torch.Size([1, 192, 14, 14]))
start = time.time()
output = m(x64)
end = time.time()
print(end-start)
