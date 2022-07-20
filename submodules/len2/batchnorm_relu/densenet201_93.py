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
        self.batchnorm2d93 = BatchNorm2d(1120, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
        self.relu93 = ReLU(inplace=True)

    def forward(self, x330):
        x331=self.batchnorm2d93(x330)
        x332=self.relu93(x331)
        return x332

m = M().eval()
x330 = torch.randn(torch.Size([1, 1120, 14, 14]))
start = time.time()
output = m(x330)
end = time.time()
print(end-start)
