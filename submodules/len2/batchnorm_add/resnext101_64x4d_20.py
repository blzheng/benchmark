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
        self.batchnorm2d57 = BatchNorm2d(1024, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)

    def forward(self, x187, x180):
        x188=self.batchnorm2d57(x187)
        x189=operator.add(x188, x180)
        return x189

m = M().eval()
x187 = torch.randn(torch.Size([1, 1024, 14, 14]))
x180 = torch.randn(torch.Size([1, 1024, 14, 14]))
start = time.time()
output = m(x187, x180)
end = time.time()
print(end-start)
