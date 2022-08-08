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
        self.conv2d35 = Conv2d(120, 30, kernel_size=(1, 1), stride=(1, 1))
        self.relu27 = ReLU()
        self.conv2d36 = Conv2d(30, 120, kernel_size=(1, 1), stride=(1, 1))
        self.sigmoid6 = Sigmoid()
        self.conv2d37 = Conv2d(120, 120, kernel_size=(1, 1), stride=(1, 1), bias=False)
        self.batchnorm2d23 = BatchNorm2d(120, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)

    def forward(self, x110, x109, x103):
        x111=self.conv2d35(x110)
        x112=self.relu27(x111)
        x113=self.conv2d36(x112)
        x114=self.sigmoid6(x113)
        x115=operator.mul(x114, x109)
        x116=self.conv2d37(x115)
        x117=self.batchnorm2d23(x116)
        x118=operator.add(x103, x117)
        return x118

m = M().eval()
x110 = torch.randn(torch.Size([1, 120, 1, 1]))
x109 = torch.randn(torch.Size([1, 120, 28, 28]))
x103 = torch.randn(torch.Size([1, 120, 28, 28]))
start = time.time()
output = m(x110, x109, x103)
end = time.time()
print(end-start)
