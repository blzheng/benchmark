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
        self.conv2d27 = Conv2d(40, 240, kernel_size=(1, 1), stride=(1, 1), bias=False)
        self.batchnorm2d27 = BatchNorm2d(240, eps=1e-05, momentum=0.00029999999999996696, affine=True, track_running_stats=True)

    def forward(self, x76, x68):
        x77=operator.add(x76, x68)
        x78=self.conv2d27(x77)
        x79=self.batchnorm2d27(x78)
        return x79

m = M().eval()
x76 = torch.randn(torch.Size([1, 40, 14, 14]))
x68 = torch.randn(torch.Size([1, 40, 14, 14]))
start = time.time()
output = m(x76, x68)
end = time.time()
print(end-start)
