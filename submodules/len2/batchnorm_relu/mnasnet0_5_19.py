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
        self.batchnorm2d28 = BatchNorm2d(240, eps=1e-05, momentum=0.00029999999999996696, affine=True, track_running_stats=True)
        self.relu19 = ReLU(inplace=True)

    def forward(self, x81):
        x82=self.batchnorm2d28(x81)
        x83=self.relu19(x82)
        return x83

m = M().eval()
x81 = torch.randn(torch.Size([1, 240, 14, 14]))
start = time.time()
output = m(x81)
end = time.time()
print(end-start)
