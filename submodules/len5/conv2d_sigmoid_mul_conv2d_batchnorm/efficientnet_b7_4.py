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
        self.conv2d20 = Conv2d(8, 192, kernel_size=(1, 1), stride=(1, 1))
        self.sigmoid4 = Sigmoid()
        self.conv2d21 = Conv2d(192, 48, kernel_size=(1, 1), stride=(1, 1), bias=False)
        self.batchnorm2d11 = BatchNorm2d(48, eps=0.001, momentum=0.01, affine=True, track_running_stats=True)

    def forward(self, x62, x59):
        x63=self.conv2d20(x62)
        x64=self.sigmoid4(x63)
        x65=operator.mul(x64, x59)
        x66=self.conv2d21(x65)
        x67=self.batchnorm2d11(x66)
        return x67

m = M().eval()
x62 = torch.randn(torch.Size([1, 8, 1, 1]))
x59 = torch.randn(torch.Size([1, 192, 56, 56]))
start = time.time()
output = m(x62, x59)
end = time.time()
print(end-start)
