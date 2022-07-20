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
        self.conv2d11 = Conv2d(8, 32, kernel_size=(1, 1), stride=(1, 1))
        self.sigmoid2 = Sigmoid()
        self.conv2d12 = Conv2d(32, 32, kernel_size=(1, 1), stride=(1, 1), bias=False)

    def forward(self, x33, x30):
        x34=self.conv2d11(x33)
        x35=self.sigmoid2(x34)
        x36=operator.mul(x35, x30)
        x37=self.conv2d12(x36)
        return x37

m = M().eval()
x33 = torch.randn(torch.Size([1, 8, 1, 1]))
x30 = torch.randn(torch.Size([1, 32, 112, 112]))
start = time.time()
output = m(x33, x30)
end = time.time()
print(end-start)
