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
        self.batchnorm2d106 = BatchNorm2d(1344, eps=0.001, momentum=0.01, affine=True, track_running_stats=True)

    def forward(self, x561):
        x562=self.batchnorm2d106(x561)
        return x562

m = M().eval()
x561 = torch.randn(torch.Size([1, 1344, 14, 14]))
start = time.time()
output = m(x561)
end = time.time()
print(end-start)
