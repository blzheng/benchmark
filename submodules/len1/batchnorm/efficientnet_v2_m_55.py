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
        self.batchnorm2d55 = BatchNorm2d(1056, eps=0.001, momentum=0.1, affine=True, track_running_stats=True)

    def forward(self, x244):
        x245=self.batchnorm2d55(x244)
        return x245

m = M().eval()
x244 = torch.randn(torch.Size([1, 1056, 14, 14]))
start = time.time()
output = m(x244)
end = time.time()
print(end-start)
