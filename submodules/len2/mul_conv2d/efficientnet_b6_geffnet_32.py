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
        self.conv2d162 = Conv2d(2064, 344, kernel_size=(1, 1), stride=(1, 1), bias=False)

    def forward(self, x477, x482):
        x483=operator.mul(x477, x482)
        x484=self.conv2d162(x483)
        return x484

m = M().eval()
x477 = torch.randn(torch.Size([1, 2064, 7, 7]))
x482 = torch.randn(torch.Size([1, 2064, 1, 1]))
start = time.time()
output = m(x477, x482)
end = time.time()
print(end-start)
