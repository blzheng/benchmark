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
        self.batchnorm2d81 = BatchNorm2d(1024, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)

    def forward(self, x267, x260):
        x268=self.batchnorm2d81(x267)
        x269=operator.add(x268, x260)
        return x269

m = M().eval()
x267 = torch.randn(torch.Size([1, 1024, 14, 14]))
x260 = torch.randn(torch.Size([1, 1024, 14, 14]))
start = time.time()
output = m(x267, x260)
end = time.time()
print(end-start)
