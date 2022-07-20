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
        self.batchnorm2d23 = BatchNorm2d(240, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)

    def forward(self, x74, x67):
        x75=self.batchnorm2d23(x74)
        x76=operator.add(x67, x75)
        return x76

m = M().eval()
x74 = torch.randn(torch.Size([1, 240, 28, 28]))
x67 = torch.randn(torch.Size([1, 240, 28, 28]))
start = time.time()
output = m(x74, x67)
end = time.time()
print(end-start)
