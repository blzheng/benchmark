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
        self.conv2d72 = Conv2d(44, 1056, kernel_size=(1, 1), stride=(1, 1))
        self.sigmoid9 = Sigmoid()

    def forward(self, x233, x230):
        x234=self.conv2d72(x233)
        x235=self.sigmoid9(x234)
        x236=operator.mul(x235, x230)
        return x236

m = M().eval()
x233 = torch.randn(torch.Size([1, 44, 1, 1]))
x230 = torch.randn(torch.Size([1, 1056, 14, 14]))
start = time.time()
output = m(x233, x230)
end = time.time()
print(end-start)
