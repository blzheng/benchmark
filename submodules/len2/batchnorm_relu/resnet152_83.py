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
        self.batchnorm2d127 = BatchNorm2d(256, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
        self.relu124 = ReLU(inplace=True)

    def forward(self, x421):
        x422=self.batchnorm2d127(x421)
        x423=self.relu124(x422)
        return x423

m = M().eval()
x421 = torch.randn(torch.Size([1, 256, 14, 14]))
start = time.time()
output = m(x421)
end = time.time()
print(end-start)
