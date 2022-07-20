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
        self.batchnorm2d21 = BatchNorm2d(160, eps=0.001, momentum=0.1, affine=True, track_running_stats=True)

    def forward(self, x86):
        x87=self.batchnorm2d21(x86)
        x88=torch.nn.functional.relu(x87,inplace=True)
        return x88

m = M().eval()
x86 = torch.randn(torch.Size([1, 160, 14, 14]))
start = time.time()
output = m(x86)
end = time.time()
print(end-start)
