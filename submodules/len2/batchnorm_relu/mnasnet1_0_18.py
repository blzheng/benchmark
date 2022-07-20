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
        self.batchnorm2d27 = BatchNorm2d(480, eps=1e-05, momentum=0.00029999999999996696, affine=True, track_running_stats=True)
        self.relu18 = ReLU(inplace=True)

    def forward(self, x78):
        x79=self.batchnorm2d27(x78)
        x80=self.relu18(x79)
        return x80

m = M().eval()
x78 = torch.randn(torch.Size([1, 480, 14, 14]))
start = time.time()
output = m(x78)
end = time.time()
print(end-start)
