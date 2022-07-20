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
        self.batchnorm2d19 = BatchNorm2d(256, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
        self.relu16 = ReLU(inplace=True)

    def forward(self, x62):
        x63=self.batchnorm2d19(x62)
        x64=self.relu16(x63)
        return x64

m = M().eval()
x62 = torch.randn(torch.Size([1, 256, 28, 28]))
start = time.time()
output = m(x62)
end = time.time()
print(end-start)
