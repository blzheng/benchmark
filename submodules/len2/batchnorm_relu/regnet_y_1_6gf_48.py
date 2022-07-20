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
        self.batchnorm2d74 = BatchNorm2d(336, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
        self.relu94 = ReLU(inplace=True)

    def forward(self, x381):
        x382=self.batchnorm2d74(x381)
        x383=self.relu94(x382)
        return x383

m = M().eval()
x381 = torch.randn(torch.Size([1, 336, 14, 14]))
start = time.time()
output = m(x381)
end = time.time()
print(end-start)