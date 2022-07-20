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
        self.batchnorm2d24 = BatchNorm2d(256, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
        self.relu21 = ReLU(inplace=True)

    def forward(self, x82, x78):
        x83=self.batchnorm2d24(x82)
        x84=operator.add(x83, x78)
        x85=self.relu21(x84)
        return x85

m = M().eval()
x82 = torch.randn(torch.Size([1, 256, 14, 14]))
x78 = torch.randn(torch.Size([1, 256, 14, 14]))
start = time.time()
output = m(x82, x78)
end = time.time()
print(end-start)
