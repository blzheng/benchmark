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
        self.conv2d53 = Conv2d(960, 240, kernel_size=(1, 1), stride=(1, 1))
        self.relu17 = ReLU()
        self.conv2d54 = Conv2d(240, 960, kernel_size=(1, 1), stride=(1, 1))
        self.hardsigmoid6 = Hardsigmoid()
        self.conv2d55 = Conv2d(960, 160, kernel_size=(1, 1), stride=(1, 1), bias=False)
        self.batchnorm2d41 = BatchNorm2d(160, eps=0.001, momentum=0.01, affine=True, track_running_stats=True)

    def forward(self, x155, x154):
        x156=self.conv2d53(x155)
        x157=self.relu17(x156)
        x158=self.conv2d54(x157)
        x159=self.hardsigmoid6(x158)
        x160=operator.mul(x159, x154)
        x161=self.conv2d55(x160)
        x162=self.batchnorm2d41(x161)
        return x162

m = M().eval()
x155 = torch.randn(torch.Size([1, 960, 1, 1]))
x154 = torch.randn(torch.Size([1, 960, 14, 14]))
start = time.time()
output = m(x155, x154)
end = time.time()
print(end-start)
