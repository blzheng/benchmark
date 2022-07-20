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
        self.conv2d69 = Conv2d(1152, 192, kernel_size=(1, 1), stride=(1, 1), bias=False)
        self.batchnorm2d41 = BatchNorm2d(192, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)

    def forward(self, x200, x196):
        x201=x200.sigmoid()
        x202=operator.mul(x196, x201)
        x203=self.conv2d69(x202)
        x204=self.batchnorm2d41(x203)
        return x204

m = M().eval()
x200 = torch.randn(torch.Size([1, 1152, 1, 1]))
x196 = torch.randn(torch.Size([1, 1152, 7, 7]))
start = time.time()
output = m(x200, x196)
end = time.time()
print(end-start)
