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
        self.batchnorm2d119 = BatchNorm2d(304, eps=0.001, momentum=0.1, affine=True, track_running_stats=True)

    def forward(self, x587):
        x588=self.batchnorm2d119(x587)
        return x588

m = M().eval()
x587 = torch.randn(torch.Size([1, 304, 7, 7]))
start = time.time()
output = m(x587)
end = time.time()
print(end-start)
