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
        self.batchnorm2d40 = BatchNorm2d(440, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)

    def forward(self, x200, x187):
        x201=self.batchnorm2d40(x200)
        x202=operator.add(x187, x201)
        return x202

m = M().eval()
x200 = torch.randn(torch.Size([1, 440, 7, 7]))
x187 = torch.randn(torch.Size([1, 440, 7, 7]))
start = time.time()
output = m(x200, x187)
end = time.time()
print(end-start)
