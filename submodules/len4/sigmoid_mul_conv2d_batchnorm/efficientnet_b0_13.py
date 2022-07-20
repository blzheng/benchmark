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
        self.sigmoid13 = Sigmoid()
        self.conv2d69 = Conv2d(1152, 192, kernel_size=(1, 1), stride=(1, 1), bias=False)
        self.batchnorm2d41 = BatchNorm2d(192, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)

    def forward(self, x206, x202):
        x207=self.sigmoid13(x206)
        x208=operator.mul(x207, x202)
        x209=self.conv2d69(x208)
        x210=self.batchnorm2d41(x209)
        return x210

m = M().eval()
x206 = torch.randn(torch.Size([1, 1152, 1, 1]))
x202 = torch.randn(torch.Size([1, 1152, 7, 7]))
start = time.time()
output = m(x206, x202)
end = time.time()
print(end-start)
