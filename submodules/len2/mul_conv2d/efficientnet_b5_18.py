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
        self.conv2d92 = Conv2d(768, 128, kernel_size=(1, 1), stride=(1, 1), bias=False)

    def forward(self, x285, x280):
        x286=operator.mul(x285, x280)
        x287=self.conv2d92(x286)
        return x287

m = M().eval()
x285 = torch.randn(torch.Size([1, 768, 1, 1]))
x280 = torch.randn(torch.Size([1, 768, 14, 14]))
start = time.time()
output = m(x285, x280)
end = time.time()
print(end-start)
