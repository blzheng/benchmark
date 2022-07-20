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
        self.batchnorm2d14 = BatchNorm2d(216, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)

    def forward(self, x68, x55):
        x69=self.batchnorm2d14(x68)
        x70=operator.add(x55, x69)
        return x70

m = M().eval()
x68 = torch.randn(torch.Size([1, 216, 28, 28]))
x55 = torch.randn(torch.Size([1, 216, 28, 28]))
start = time.time()
output = m(x68, x55)
end = time.time()
print(end-start)
