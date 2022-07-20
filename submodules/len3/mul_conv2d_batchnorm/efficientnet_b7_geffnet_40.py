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
        self.conv2d201 = Conv2d(2304, 384, kernel_size=(1, 1), stride=(1, 1), bias=False)
        self.batchnorm2d119 = BatchNorm2d(384, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)

    def forward(self, x595, x600):
        x601=operator.mul(x595, x600)
        x602=self.conv2d201(x601)
        x603=self.batchnorm2d119(x602)
        return x603

m = M().eval()
x595 = torch.randn(torch.Size([1, 2304, 7, 7]))
x600 = torch.randn(torch.Size([1, 2304, 1, 1]))
start = time.time()
output = m(x595, x600)
end = time.time()
print(end-start)
