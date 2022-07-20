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
        self.conv2d7 = Conv2d(8, 32, kernel_size=(1, 1), stride=(1, 1))
        self.sigmoid1 = Sigmoid()

    def forward(self, x20, x17):
        x21=self.conv2d7(x20)
        x22=self.sigmoid1(x21)
        x23=operator.mul(x22, x17)
        return x23

m = M().eval()
x20 = torch.randn(torch.Size([1, 8, 1, 1]))
x17 = torch.randn(torch.Size([1, 32, 112, 112]))
start = time.time()
output = m(x20, x17)
end = time.time()
print(end-start)
