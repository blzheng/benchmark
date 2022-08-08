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
        self.batchnorm2d51 = BatchNorm2d(1024, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)

    def forward(self, x169, x162):
        x170=self.batchnorm2d51(x169)
        x171=operator.add(x170, x162)
        return x171

m = M().eval()
x169 = torch.randn(torch.Size([1, 1024, 28, 28]))
x162 = torch.randn(torch.Size([1, 1024, 28, 28]))
start = time.time()
output = m(x169, x162)
end = time.time()
print(end-start)