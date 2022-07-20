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
        self.batchnorm2d33 = BatchNorm2d(1024, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)

    def forward(self, x107, x100):
        x108=self.batchnorm2d33(x107)
        x109=operator.add(x108, x100)
        return x109

m = M().eval()
x107 = torch.randn(torch.Size([1, 1024, 14, 14]))
x100 = torch.randn(torch.Size([1, 1024, 14, 14]))
start = time.time()
output = m(x107, x100)
end = time.time()
print(end-start)