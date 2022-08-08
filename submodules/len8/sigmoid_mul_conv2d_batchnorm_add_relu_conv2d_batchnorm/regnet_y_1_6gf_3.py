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
        self.sigmoid3 = Sigmoid()
        self.conv2d22 = Conv2d(120, 120, kernel_size=(1, 1), stride=(1, 1), bias=False)
        self.batchnorm2d14 = BatchNorm2d(120, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
        self.relu16 = ReLU(inplace=True)
        self.conv2d23 = Conv2d(120, 120, kernel_size=(1, 1), stride=(1, 1), bias=False)
        self.batchnorm2d15 = BatchNorm2d(120, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)

    def forward(self, x65, x61, x55):
        x66=self.sigmoid3(x65)
        x67=operator.mul(x66, x61)
        x68=self.conv2d22(x67)
        x69=self.batchnorm2d14(x68)
        x70=operator.add(x55, x69)
        x71=self.relu16(x70)
        x72=self.conv2d23(x71)
        x73=self.batchnorm2d15(x72)
        return x73

m = M().eval()
x65 = torch.randn(torch.Size([1, 120, 1, 1]))
x61 = torch.randn(torch.Size([1, 120, 28, 28]))
x55 = torch.randn(torch.Size([1, 120, 28, 28]))
start = time.time()
output = m(x65, x61, x55)
end = time.time()
print(end-start)
