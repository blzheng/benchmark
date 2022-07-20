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
        self.batchnorm2d33 = BatchNorm2d(240, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
        self.relu22 = ReLU(inplace=True)

    def forward(self, x108):
        x109=self.batchnorm2d33(x108)
        x110=self.relu22(x109)
        return x110

m = M().eval()
x108 = torch.randn(torch.Size([1, 240, 14, 14]))
start = time.time()
output = m(x108)
end = time.time()
print(end-start)
