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
        self.conv2d227 = Conv2d(384, 2304, kernel_size=(1, 1), stride=(1, 1), bias=False)

    def forward(self, x678, x664):
        x679=operator.add(x678, x664)
        x680=self.conv2d227(x679)
        return x680

m = M().eval()
x678 = torch.randn(torch.Size([1, 384, 7, 7]))
x664 = torch.randn(torch.Size([1, 384, 7, 7]))
start = time.time()
output = m(x678, x664)
end = time.time()
print(end-start)
