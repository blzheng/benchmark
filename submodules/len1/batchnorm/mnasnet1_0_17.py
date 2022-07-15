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
        self.batchnorm2d17 = BatchNorm2d(40, eps=1e-05, momentum=0.00029999999999996696, affine=True, track_running_stats=True)

    def forward(self, x49):
        x50=self.batchnorm2d17(x49)
        return x50

m = M().eval()
x49 = torch.randn(torch.Size([1, 40, 28, 28]))
start = time.time()
output = m(x49)
end = time.time()
print(end-start)
