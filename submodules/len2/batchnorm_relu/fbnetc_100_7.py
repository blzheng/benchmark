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
        self.batchnorm2d10 = BatchNorm2d(24, eps=0.001, momentum=0.1, affine=True, track_running_stats=True)
        self.relu7 = ReLU(inplace=True)

    def forward(self, x33):
        x34=self.batchnorm2d10(x33)
        x35=self.relu7(x34)
        return x35

m = M().eval()
x33 = torch.randn(torch.Size([1, 24, 56, 56]))
start = time.time()
output = m(x33)
end = time.time()
print(end-start)
