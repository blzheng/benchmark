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
        self.batchnorm2d4 = BatchNorm2d(80, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
        self.relu3 = ReLU(inplace=True)

    def forward(self, x12, x5):
        x13=self.batchnorm2d4(x12)
        x14=operator.add(x5, x13)
        x15=self.relu3(x14)
        return x15

m = M().eval()
x12 = torch.randn(torch.Size([1, 80, 56, 56]))
x5 = torch.randn(torch.Size([1, 80, 56, 56]))
start = time.time()
output = m(x12, x5)
end = time.time()
print(end-start)
