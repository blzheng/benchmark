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
        self.conv2d123 = Conv2d(1632, 272, kernel_size=(1, 1), stride=(1, 1), bias=False)

    def forward(self, x359, x364):
        x365=operator.mul(x359, x364)
        x366=self.conv2d123(x365)
        return x366

m = M().eval()
x359 = torch.randn(torch.Size([1, 1632, 7, 7]))
x364 = torch.randn(torch.Size([1, 1632, 1, 1]))
start = time.time()
output = m(x359, x364)
end = time.time()
print(end-start)
