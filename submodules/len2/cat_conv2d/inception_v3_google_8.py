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
        self.conv2d70 = Conv2d(768, 192, kernel_size=(1, 1), stride=(1, 1), bias=False)

    def forward(self, x212, x221, x236, x240):
        x241=torch.cat([x212, x221, x236, x240], 1)
        x242=self.conv2d70(x241)
        return x242

m = M().eval()
x212 = torch.randn(torch.Size([1, 192, 12, 12]))
x221 = torch.randn(torch.Size([1, 192, 12, 12]))
x236 = torch.randn(torch.Size([1, 192, 12, 12]))
x240 = torch.randn(torch.Size([1, 192, 12, 12]))
start = time.time()
output = m(x212, x221, x236, x240)
end = time.time()
print(end-start)
