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
        self.batchnorm2d21 = BatchNorm2d(288, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)

    def forward(self, x66, x59):
        x67=self.batchnorm2d21(x66)
        x68=operator.add(x59, x67)
        return x68

m = M().eval()
x66 = torch.randn(torch.Size([1, 288, 14, 14]))
x59 = torch.randn(torch.Size([1, 288, 14, 14]))
start = time.time()
output = m(x66, x59)
end = time.time()
print(end-start)
