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
        self.batchnorm2d31 = BatchNorm2d(128, eps=0.001, momentum=0.1, affine=True, track_running_stats=True)

    def forward(self, x117):
        x118=self.batchnorm2d31(x117)
        x119=torch.nn.functional.relu(x118,inplace=True)
        return x119

m = M().eval()
x117 = torch.randn(torch.Size([1, 128, 12, 12]))
start = time.time()
output = m(x117)
end = time.time()
print(end-start)
