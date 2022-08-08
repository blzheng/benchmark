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
        self.batchnorm2d49 = BatchNorm2d(256, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
        self.relu22 = ReLU()

    def forward(self, x193):
        x194=self.batchnorm2d49(x193)
        x195=self.relu22(x194)
        return x195

m = M().eval()
x193 = torch.randn(torch.Size([1, 256, 14, 14]))
start = time.time()
output = m(x193)
end = time.time()
print(end-start)
