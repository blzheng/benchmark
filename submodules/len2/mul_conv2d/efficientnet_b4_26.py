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
        self.conv2d133 = Conv2d(1632, 272, kernel_size=(1, 1), stride=(1, 1), bias=False)

    def forward(self, x412, x407):
        x413=operator.mul(x412, x407)
        x414=self.conv2d133(x413)
        return x414

m = M().eval()
x412 = torch.randn(torch.Size([1, 1632, 1, 1]))
x407 = torch.randn(torch.Size([1, 1632, 7, 7]))
start = time.time()
output = m(x412, x407)
end = time.time()
print(end-start)
