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
        self.batchnorm2d3 = BatchNorm2d(128, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
        self.relu3 = ReLU(inplace=True)

    def forward(self, x11):
        x12=self.batchnorm2d3(x11)
        x13=self.relu3(x12)
        return x13

m = M().eval()
x11 = torch.randn(torch.Size([1, 128, 112, 112]))
start = time.time()
output = m(x11)
end = time.time()
print(end-start)