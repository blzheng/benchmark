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
        self.conv2d52 = Conv2d(20, 480, kernel_size=(1, 1), stride=(1, 1))
        self.sigmoid10 = Sigmoid()
        self.conv2d53 = Conv2d(480, 80, kernel_size=(1, 1), stride=(1, 1), bias=False)

    def forward(self, x158, x155):
        x159=self.conv2d52(x158)
        x160=self.sigmoid10(x159)
        x161=operator.mul(x160, x155)
        x162=self.conv2d53(x161)
        return x162

m = M().eval()
x158 = torch.randn(torch.Size([1, 20, 1, 1]))
x155 = torch.randn(torch.Size([1, 480, 14, 14]))
start = time.time()
output = m(x158, x155)
end = time.time()
print(end-start)
