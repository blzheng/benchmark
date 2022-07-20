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
        self.conv2d73 = Conv2d(110, 440, kernel_size=(1, 1), stride=(1, 1))
        self.sigmoid13 = Sigmoid()
        self.conv2d74 = Conv2d(440, 440, kernel_size=(1, 1), stride=(1, 1), bias=False)

    def forward(self, x228, x225):
        x229=self.conv2d73(x228)
        x230=self.sigmoid13(x229)
        x231=operator.mul(x230, x225)
        x232=self.conv2d74(x231)
        return x232

m = M().eval()
x228 = torch.randn(torch.Size([1, 110, 1, 1]))
x225 = torch.randn(torch.Size([1, 440, 7, 7]))
start = time.time()
output = m(x228, x225)
end = time.time()
print(end-start)