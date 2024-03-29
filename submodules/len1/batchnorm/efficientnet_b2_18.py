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
        self.batchnorm2d18 = BatchNorm2d(288, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)

    def forward(self, x91):
        x92=self.batchnorm2d18(x91)
        return x92

m = M().eval()
x91 = torch.randn(torch.Size([1, 288, 28, 28]))
start = time.time()
output = m(x91)
end = time.time()
print(end-start)
