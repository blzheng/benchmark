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
        self.batchnorm2d93 = BatchNorm2d(1024, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
        self.relu88 = ReLU(inplace=True)

    def forward(self, x309, x302):
        x310=self.batchnorm2d93(x309)
        x311=operator.add(x310, x302)
        x312=self.relu88(x311)
        return x312

m = M().eval()
x309 = torch.randn(torch.Size([1, 1024, 28, 28]))
x302 = torch.randn(torch.Size([1, 1024, 28, 28]))
start = time.time()
output = m(x309, x302)
end = time.time()
print(end-start)