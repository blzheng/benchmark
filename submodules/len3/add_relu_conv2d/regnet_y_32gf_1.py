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
        self.relu8 = ReLU(inplace=True)
        self.conv2d12 = Conv2d(232, 696, kernel_size=(1, 1), stride=(2, 2), bias=False)

    def forward(self, x21, x35):
        x36=operator.add(x21, x35)
        x37=self.relu8(x36)
        x38=self.conv2d12(x37)
        return x38

m = M().eval()
x21 = torch.randn(torch.Size([1, 232, 56, 56]))
x35 = torch.randn(torch.Size([1, 232, 56, 56]))
start = time.time()
output = m(x21, x35)
end = time.time()
print(end-start)
