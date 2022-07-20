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
        self.batchnorm2d66 = BatchNorm2d(144, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)

    def forward(self, x336, x323):
        x337=self.batchnorm2d66(x336)
        x338=operator.add(x337, x323)
        return x338

m = M().eval()
x336 = torch.randn(torch.Size([1, 144, 14, 14]))
x323 = torch.randn(torch.Size([1, 144, 14, 14]))
start = time.time()
output = m(x336, x323)
end = time.time()
print(end-start)
