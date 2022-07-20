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
        self.conv2d61 = Conv2d(480, 80, kernel_size=(1, 1), stride=(1, 1), bias=False)

    def forward(self, x182, x178):
        x183=x182.sigmoid()
        x184=operator.mul(x178, x183)
        x185=self.conv2d61(x184)
        return x185

m = M().eval()
x182 = torch.randn(torch.Size([1, 480, 1, 1]))
x178 = torch.randn(torch.Size([1, 480, 28, 28]))
start = time.time()
output = m(x182, x178)
end = time.time()
print(end-start)
