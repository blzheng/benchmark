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
        self.conv2d138 = Conv2d(1632, 272, kernel_size=(1, 1), stride=(1, 1), bias=False)

    def forward(self, x408, x404):
        x409=x408.sigmoid()
        x410=operator.mul(x404, x409)
        x411=self.conv2d138(x410)
        return x411

m = M().eval()
x408 = torch.randn(torch.Size([1, 1632, 1, 1]))
x404 = torch.randn(torch.Size([1, 1632, 7, 7]))
start = time.time()
output = m(x408, x404)
end = time.time()
print(end-start)
