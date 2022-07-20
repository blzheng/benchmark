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
        self.batchnorm2d33 = BatchNorm2d(896, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)

    def forward(self, x166, x153):
        x167=self.batchnorm2d33(x166)
        x168=operator.add(x153, x167)
        return x168

m = M().eval()
x166 = torch.randn(torch.Size([1, 896, 14, 14]))
x153 = torch.randn(torch.Size([1, 896, 14, 14]))
start = time.time()
output = m(x166, x153)
end = time.time()
print(end-start)
