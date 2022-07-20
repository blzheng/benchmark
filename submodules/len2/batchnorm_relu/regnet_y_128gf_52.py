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
        self.batchnorm2d80 = BatchNorm2d(2904, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
        self.relu102 = ReLU(inplace=True)

    def forward(self, x413):
        x414=self.batchnorm2d80(x413)
        x415=self.relu102(x414)
        return x415

m = M().eval()
x413 = torch.randn(torch.Size([1, 2904, 14, 14]))
start = time.time()
output = m(x413)
end = time.time()
print(end-start)
