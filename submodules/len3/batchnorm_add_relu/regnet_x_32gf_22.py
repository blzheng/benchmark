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
        self.batchnorm2d63 = BatchNorm2d(1344, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
        self.relu60 = ReLU(inplace=True)

    def forward(self, x206, x199):
        x207=self.batchnorm2d63(x206)
        x208=operator.add(x199, x207)
        x209=self.relu60(x208)
        return x209

m = M().eval()
x206 = torch.randn(torch.Size([1, 1344, 14, 14]))
x199 = torch.randn(torch.Size([1, 1344, 14, 14]))
start = time.time()
output = m(x206, x199)
end = time.time()
print(end-start)
