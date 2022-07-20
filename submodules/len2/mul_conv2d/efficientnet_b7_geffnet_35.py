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
        self.conv2d176 = Conv2d(1344, 224, kernel_size=(1, 1), stride=(1, 1), bias=False)

    def forward(self, x521, x526):
        x527=operator.mul(x521, x526)
        x528=self.conv2d176(x527)
        return x528

m = M().eval()
x521 = torch.randn(torch.Size([1, 1344, 14, 14]))
x526 = torch.randn(torch.Size([1, 1344, 1, 1]))
start = time.time()
output = m(x521, x526)
end = time.time()
print(end-start)
