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
        self.avgpool2d1 = AvgPool2d(kernel_size=2, stride=2, padding=0)
        self.batchnorm2d41 = BatchNorm2d(288, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)

    def forward(self, x139, x147):
        x140=self.avgpool2d1(x139)
        x148=torch.cat([x140, x147], 1)
        x149=self.batchnorm2d41(x148)
        return x149

m = M().eval()
x139 = torch.randn(torch.Size([1, 256, 28, 28]))
x147 = torch.randn(torch.Size([1, 32, 14, 14]))
start = time.time()
output = m(x139, x147)
end = time.time()
print(end-start)
