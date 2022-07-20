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
        self.conv2d65 = Conv2d(192, 1152, kernel_size=(1, 1), stride=(1, 1), bias=False)

    def forward(self, x195, x180):
        x196=operator.add(x195, x180)
        x197=self.conv2d65(x196)
        return x197

m = M().eval()
x195 = torch.randn(torch.Size([1, 192, 7, 7]))
x180 = torch.randn(torch.Size([1, 192, 7, 7]))
start = time.time()
output = m(x195, x180)
end = time.time()
print(end-start)
