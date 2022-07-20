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
        self.batchnorm2d30 = BatchNorm2d(336, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
        self.relu36 = ReLU(inplace=True)

    def forward(self, x150, x137):
        x151=self.batchnorm2d30(x150)
        x152=operator.add(x137, x151)
        x153=self.relu36(x152)
        return x153

m = M().eval()
x150 = torch.randn(torch.Size([1, 336, 14, 14]))
x137 = torch.randn(torch.Size([1, 336, 14, 14]))
start = time.time()
output = m(x150, x137)
end = time.time()
print(end-start)
