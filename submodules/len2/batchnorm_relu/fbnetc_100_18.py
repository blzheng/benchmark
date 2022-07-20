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
        self.batchnorm2d26 = BatchNorm2d(192, eps=0.001, momentum=0.1, affine=True, track_running_stats=True)
        self.relu18 = ReLU(inplace=True)

    def forward(self, x85):
        x86=self.batchnorm2d26(x85)
        x87=self.relu18(x86)
        return x87

m = M().eval()
x85 = torch.randn(torch.Size([1, 192, 14, 14]))
start = time.time()
output = m(x85)
end = time.time()
print(end-start)
