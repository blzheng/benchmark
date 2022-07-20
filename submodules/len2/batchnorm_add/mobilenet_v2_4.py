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
        self.batchnorm2d38 = BatchNorm2d(96, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)

    def forward(self, x110, x103):
        x111=self.batchnorm2d38(x110)
        x112=operator.add(x103, x111)
        return x112

m = M().eval()
x110 = torch.randn(torch.Size([1, 96, 14, 14]))
x103 = torch.randn(torch.Size([1, 96, 14, 14]))
start = time.time()
output = m(x110, x103)
end = time.time()
print(end-start)
