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
        self.conv2d61 = Conv2d(48, 768, kernel_size=(1, 1), stride=(1, 1))
        self.sigmoid5 = Sigmoid()
        self.conv2d62 = Conv2d(768, 192, kernel_size=(1, 1), stride=(1, 1), bias=False)

    def forward(self, x204, x201):
        x205=self.conv2d61(x204)
        x206=self.sigmoid5(x205)
        x207=operator.mul(x206, x201)
        x208=self.conv2d62(x207)
        return x208

m = M().eval()
x204 = torch.randn(torch.Size([1, 48, 1, 1]))
x201 = torch.randn(torch.Size([1, 768, 14, 14]))
start = time.time()
output = m(x204, x201)
end = time.time()
print(end-start)
