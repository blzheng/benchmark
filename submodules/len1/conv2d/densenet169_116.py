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
        self.conv2d116 = Conv2d(832, 128, kernel_size=(1, 1), stride=(1, 1), bias=False)

    def forward(self, x414):
        x415=self.conv2d116(x414)
        return x415

m = M().eval()
x414 = torch.randn(torch.Size([1, 832, 7, 7]))
start = time.time()
output = m(x414)
end = time.time()
print(end-start)
