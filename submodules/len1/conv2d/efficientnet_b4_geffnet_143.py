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
        self.conv2d143 = Conv2d(1632, 272, kernel_size=(1, 1), stride=(1, 1), bias=False)

    def forward(self, x425):
        x426=self.conv2d143(x425)
        return x426

m = M().eval()
x425 = torch.randn(torch.Size([1, 1632, 7, 7]))
start = time.time()
output = m(x425)
end = time.time()
print(end-start)
