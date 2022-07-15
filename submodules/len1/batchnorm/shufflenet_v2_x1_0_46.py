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
        self.batchnorm2d46 = BatchNorm2d(232, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)

    def forward(self, x300):
        x301=self.batchnorm2d46(x300)
        return x301

m = M().eval()
x300 = torch.randn(torch.Size([1, 232, 7, 7]))
start = time.time()
output = m(x300)
end = time.time()
print(end-start)
