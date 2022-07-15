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
        self.batchnorm2d95 = BatchNorm2d(304, eps=0.001, momentum=0.1, affine=True, track_running_stats=True)

    def forward(self, x459):
        x460=self.batchnorm2d95(x459)
        return x460

m = M().eval()
x459 = torch.randn(torch.Size([1, 304, 7, 7]))
start = time.time()
output = m(x459)
end = time.time()
print(end-start)
