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
        self.batchnorm2d41 = BatchNorm2d(720, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
        self.relu38 = ReLU(inplace=True)

    def forward(self, x133):
        x134=self.batchnorm2d41(x133)
        x135=self.relu38(x134)
        return x135

m = M().eval()
x133 = torch.randn(torch.Size([1, 720, 14, 14]))
start = time.time()
output = m(x133)
end = time.time()
print(end-start)
