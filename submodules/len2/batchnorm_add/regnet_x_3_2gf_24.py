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
        self.batchnorm2d69 = BatchNorm2d(432, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)

    def forward(self, x226, x219):
        x227=self.batchnorm2d69(x226)
        x228=operator.add(x219, x227)
        return x228

m = M().eval()
x226 = torch.randn(torch.Size([1, 432, 14, 14]))
x219 = torch.randn(torch.Size([1, 432, 14, 14]))
start = time.time()
output = m(x226, x219)
end = time.time()
print(end-start)
