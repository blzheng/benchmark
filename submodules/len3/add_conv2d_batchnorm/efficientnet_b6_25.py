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
        self.conv2d153 = Conv2d(200, 1200, kernel_size=(1, 1), stride=(1, 1), bias=False)
        self.batchnorm2d91 = BatchNorm2d(1200, eps=0.001, momentum=0.01, affine=True, track_running_stats=True)

    def forward(self, x479, x464):
        x480=operator.add(x479, x464)
        x481=self.conv2d153(x480)
        x482=self.batchnorm2d91(x481)
        return x482

m = M().eval()
x479 = torch.randn(torch.Size([1, 200, 14, 14]))
x464 = torch.randn(torch.Size([1, 200, 14, 14]))
start = time.time()
output = m(x479, x464)
end = time.time()
print(end-start)
