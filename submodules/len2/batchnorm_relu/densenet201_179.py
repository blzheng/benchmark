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
        self.batchnorm2d179 = BatchNorm2d(128, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
        self.relu179 = ReLU(inplace=True)

    def forward(self, x632):
        x633=self.batchnorm2d179(x632)
        x634=self.relu179(x633)
        return x634

m = M().eval()
x632 = torch.randn(torch.Size([1, 128, 7, 7]))
start = time.time()
output = m(x632)
end = time.time()
print(end-start)
