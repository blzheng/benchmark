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
        self.conv2d23 = Conv2d(240, 64, kernel_size=(1, 1), stride=(1, 1))
        self.relu8 = ReLU()
        self.conv2d24 = Conv2d(64, 240, kernel_size=(1, 1), stride=(1, 1))
        self.hardsigmoid3 = Hardsigmoid()
        self.conv2d25 = Conv2d(240, 40, kernel_size=(1, 1), stride=(1, 1), bias=False)
        self.batchnorm2d17 = BatchNorm2d(40, eps=0.001, momentum=0.01, affine=True, track_running_stats=True)

    def forward(self, x67, x66):
        x68=self.conv2d23(x67)
        x69=self.relu8(x68)
        x70=self.conv2d24(x69)
        x71=self.hardsigmoid3(x70)
        x72=operator.mul(x71, x66)
        x73=self.conv2d25(x72)
        x74=self.batchnorm2d17(x73)
        return x74

m = M().eval()
x67 = torch.randn(torch.Size([1, 240, 1, 1]))
x66 = torch.randn(torch.Size([1, 240, 14, 14]))
start = time.time()
output = m(x67, x66)
end = time.time()
print(end-start)
