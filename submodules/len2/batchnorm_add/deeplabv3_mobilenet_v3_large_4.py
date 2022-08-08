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
        self.batchnorm2d29 = BatchNorm2d(80, eps=0.001, momentum=0.01, affine=True, track_running_stats=True)

    def forward(self, x105, x98):
        x106=self.batchnorm2d29(x105)
        x107=operator.add(x106, x98)
        return x107

m = M().eval()
x105 = torch.randn(torch.Size([1, 80, 14, 14]))
x98 = torch.randn(torch.Size([1, 80, 14, 14]))
start = time.time()
output = m(x105, x98)
end = time.time()
print(end-start)
