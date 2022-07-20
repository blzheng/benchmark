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
        self.conv2d13 = Conv2d(6, 144, kernel_size=(1, 1), stride=(1, 1))
        self.sigmoid2 = Sigmoid()

    def forward(self, x37, x34):
        x38=self.conv2d13(x37)
        x39=self.sigmoid2(x38)
        x40=operator.mul(x39, x34)
        return x40

m = M().eval()
x37 = torch.randn(torch.Size([1, 6, 1, 1]))
x34 = torch.randn(torch.Size([1, 144, 56, 56]))
start = time.time()
output = m(x37, x34)
end = time.time()
print(end-start)
