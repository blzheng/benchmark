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
        self.batchnorm2d26 = BatchNorm2d(320, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)

    def forward(self, x51, x58, x65, x72, x79, x86, x93):
        x94=torch.cat([x51, x58, x65, x72, x79, x86, x93], 1)
        x95=self.batchnorm2d26(x94)
        return x95

m = M().eval()
x51 = torch.randn(torch.Size([1, 128, 28, 28]))
x58 = torch.randn(torch.Size([1, 32, 28, 28]))
x65 = torch.randn(torch.Size([1, 32, 28, 28]))
x72 = torch.randn(torch.Size([1, 32, 28, 28]))
x79 = torch.randn(torch.Size([1, 32, 28, 28]))
x86 = torch.randn(torch.Size([1, 32, 28, 28]))
x93 = torch.randn(torch.Size([1, 32, 28, 28]))
start = time.time()
output = m(x51, x58, x65, x72, x79, x86, x93)
end = time.time()
print(end-start)
