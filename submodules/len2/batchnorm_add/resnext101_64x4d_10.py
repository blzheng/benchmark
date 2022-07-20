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
        self.batchnorm2d27 = BatchNorm2d(1024, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)

    def forward(self, x87, x86):
        x88=self.batchnorm2d27(x87)
        x89=operator.add(x86, x88)
        return x89

m = M().eval()
x87 = torch.randn(torch.Size([1, 1024, 14, 14]))
x86 = torch.randn(torch.Size([1, 1024, 14, 14]))
start = time.time()
output = m(x87, x86)
end = time.time()
print(end-start)
