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
        self.conv2d118 = Conv2d(1632, 272, kernel_size=(1, 1), stride=(1, 1), bias=False)

    def forward(self, x348, x344):
        x349=x348.sigmoid()
        x350=operator.mul(x344, x349)
        x351=self.conv2d118(x350)
        return x351

m = M().eval()
x348 = torch.randn(torch.Size([1, 1632, 1, 1]))
x344 = torch.randn(torch.Size([1, 1632, 7, 7]))
start = time.time()
output = m(x348, x344)
end = time.time()
print(end-start)
