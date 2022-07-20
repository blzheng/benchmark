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
        self.batchnorm2d11 = BatchNorm2d(24, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)

    def forward(self, x37, x29):
        x38=self.batchnorm2d11(x37)
        x39=operator.add(x38, x29)
        return x39

m = M().eval()
x37 = torch.randn(torch.Size([1, 24, 56, 56]))
x29 = torch.randn(torch.Size([1, 24, 56, 56]))
start = time.time()
output = m(x37, x29)
end = time.time()
print(end-start)
