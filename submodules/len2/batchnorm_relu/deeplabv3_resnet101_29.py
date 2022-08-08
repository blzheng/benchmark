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
        self.batchnorm2d46 = BatchNorm2d(256, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
        self.relu43 = ReLU(inplace=True)

    def forward(self, x153):
        x154=self.batchnorm2d46(x153)
        x155=self.relu43(x154)
        return x155

m = M().eval()
x153 = torch.randn(torch.Size([1, 256, 28, 28]))
start = time.time()
output = m(x153)
end = time.time()
print(end-start)
