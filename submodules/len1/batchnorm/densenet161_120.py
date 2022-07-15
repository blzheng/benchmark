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
        self.batchnorm2d120 = BatchNorm2d(1248, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)

    def forward(self, x426):
        x427=self.batchnorm2d120(x426)
        return x427

m = M().eval()
x426 = torch.randn(torch.Size([1, 1248, 7, 7]))
start = time.time()
output = m(x426)
end = time.time()
print(end-start)
