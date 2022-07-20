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
        self.batchnorm2d17 = BatchNorm2d(512, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)

    def forward(self, x55, x48):
        x56=self.batchnorm2d17(x55)
        x57=operator.add(x56, x48)
        return x57

m = M().eval()
x55 = torch.randn(torch.Size([1, 512, 28, 28]))
x48 = torch.randn(torch.Size([1, 512, 28, 28]))
start = time.time()
output = m(x55, x48)
end = time.time()
print(end-start)
