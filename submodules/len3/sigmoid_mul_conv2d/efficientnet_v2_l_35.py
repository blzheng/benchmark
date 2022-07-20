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
        self.sigmoid35 = Sigmoid()
        self.conv2d212 = Conv2d(2304, 384, kernel_size=(1, 1), stride=(1, 1), bias=False)

    def forward(self, x681, x677):
        x682=self.sigmoid35(x681)
        x683=operator.mul(x682, x677)
        x684=self.conv2d212(x683)
        return x684

m = M().eval()
x681 = torch.randn(torch.Size([1, 2304, 1, 1]))
x677 = torch.randn(torch.Size([1, 2304, 7, 7]))
start = time.time()
output = m(x681, x677)
end = time.time()
print(end-start)
