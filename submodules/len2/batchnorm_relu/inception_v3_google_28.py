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
        self.batchnorm2d28 = BatchNorm2d(96, eps=0.001, momentum=0.1, affine=True, track_running_stats=True)

    def forward(self, x106):
        x107=self.batchnorm2d28(x106)
        x108=torch.nn.functional.relu(x107,inplace=True)
        return x108

m = M().eval()
x106 = torch.randn(torch.Size([1, 96, 25, 25]))
start = time.time()
output = m(x106)
end = time.time()
print(end-start)
