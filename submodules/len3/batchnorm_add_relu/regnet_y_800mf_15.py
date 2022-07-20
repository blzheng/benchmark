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
        self.batchnorm2d40 = BatchNorm2d(784, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
        self.relu52 = ReLU(inplace=True)

    def forward(self, x202, x217):
        x203=self.batchnorm2d40(x202)
        x218=operator.add(x203, x217)
        x219=self.relu52(x218)
        return x219

m = M().eval()
x202 = torch.randn(torch.Size([1, 784, 7, 7]))
x217 = torch.randn(torch.Size([1, 784, 7, 7]))
start = time.time()
output = m(x202, x217)
end = time.time()
print(end-start)
