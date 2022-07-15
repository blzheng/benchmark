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
        self.batchnorm2d39 = BatchNorm2d(1152, eps=1e-05, momentum=0.00029999999999996696, affine=True, track_running_stats=True)

    def forward(self, x112):
        x113=self.batchnorm2d39(x112)
        return x113

m = M().eval()
x112 = torch.randn(torch.Size([1, 1152, 7, 7]))
start = time.time()
output = m(x112)
end = time.time()
print(end-start)
