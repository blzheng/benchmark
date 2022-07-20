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
        self.batchnorm2d60 = BatchNorm2d(192, eps=0.001, momentum=0.1, affine=True, track_running_stats=True)

    def forward(self, x210):
        x211=self.batchnorm2d60(x210)
        x212=torch.nn.functional.relu(x211,inplace=True)
        return x212

m = M().eval()
x210 = torch.randn(torch.Size([1, 192, 12, 12]))
start = time.time()
output = m(x210)
end = time.time()
print(end-start)
