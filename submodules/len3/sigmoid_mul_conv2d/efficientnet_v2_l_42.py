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
        self.sigmoid42 = Sigmoid()
        self.conv2d247 = Conv2d(2304, 384, kernel_size=(1, 1), stride=(1, 1), bias=False)

    def forward(self, x793, x789):
        x794=self.sigmoid42(x793)
        x795=operator.mul(x794, x789)
        x796=self.conv2d247(x795)
        return x796

m = M().eval()
x793 = torch.randn(torch.Size([1, 2304, 1, 1]))
x789 = torch.randn(torch.Size([1, 2304, 7, 7]))
start = time.time()
output = m(x793, x789)
end = time.time()
print(end-start)
