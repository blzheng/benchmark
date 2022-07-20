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
        self.batchnorm2d47 = BatchNorm2d(336, eps=0.001, momentum=0.1, affine=True, track_running_stats=True)
        self.relu32 = ReLU(inplace=True)

    def forward(self, x153):
        x154=self.batchnorm2d47(x153)
        x155=self.relu32(x154)
        return x155

m = M().eval()
x153 = torch.randn(torch.Size([1, 336, 14, 14]))
start = time.time()
output = m(x153)
end = time.time()
print(end-start)
