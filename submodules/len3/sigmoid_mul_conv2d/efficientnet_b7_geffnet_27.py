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
        self.conv2d136 = Conv2d(960, 160, kernel_size=(1, 1), stride=(1, 1), bias=False)

    def forward(self, x406, x402):
        x407=x406.sigmoid()
        x408=operator.mul(x402, x407)
        x409=self.conv2d136(x408)
        return x409

m = M().eval()
x406 = torch.randn(torch.Size([1, 960, 1, 1]))
x402 = torch.randn(torch.Size([1, 960, 14, 14]))
start = time.time()
output = m(x406, x402)
end = time.time()
print(end-start)
