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
        self.batchnorm2d24 = BatchNorm2d(408, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)

    def forward(self, x76, x69):
        x77=self.batchnorm2d24(x76)
        x78=operator.add(x69, x77)
        return x78

m = M().eval()
x76 = torch.randn(torch.Size([1, 408, 14, 14]))
x69 = torch.randn(torch.Size([1, 408, 14, 14]))
start = time.time()
output = m(x76, x69)
end = time.time()
print(end-start)
