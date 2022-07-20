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
        self.batchnorm2d28 = BatchNorm2d(288, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
        self.relu25 = ReLU(inplace=True)

    def forward(self, x90):
        x91=self.batchnorm2d28(x90)
        x92=self.relu25(x91)
        return x92

m = M().eval()
x90 = torch.randn(torch.Size([1, 288, 14, 14]))
start = time.time()
output = m(x90)
end = time.time()
print(end-start)
