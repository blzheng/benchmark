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
        self.batchnorm2d30 = BatchNorm2d(480, eps=0.001, momentum=0.01, affine=True, track_running_stats=True)

    def forward(self, x106):
        x107=self.batchnorm2d30(x106)
        return x107

m = M().eval()
x106 = torch.randn(torch.Size([1, 480, 14, 14]))
start = time.time()
output = m(x106)
end = time.time()
print(end-start)
