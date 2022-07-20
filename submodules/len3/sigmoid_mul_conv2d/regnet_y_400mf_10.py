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
        self.sigmoid10 = Sigmoid()
        self.conv2d59 = Conv2d(440, 440, kernel_size=(1, 1), stride=(1, 1), bias=False)

    def forward(self, x181, x177):
        x182=self.sigmoid10(x181)
        x183=operator.mul(x182, x177)
        x184=self.conv2d59(x183)
        return x184

m = M().eval()
x181 = torch.randn(torch.Size([1, 440, 1, 1]))
x177 = torch.randn(torch.Size([1, 440, 7, 7]))
start = time.time()
output = m(x181, x177)
end = time.time()
print(end-start)
