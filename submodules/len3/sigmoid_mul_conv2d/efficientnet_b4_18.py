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
        self.sigmoid18 = Sigmoid()
        self.conv2d93 = Conv2d(960, 160, kernel_size=(1, 1), stride=(1, 1), bias=False)

    def forward(self, x285, x281):
        x286=self.sigmoid18(x285)
        x287=operator.mul(x286, x281)
        x288=self.conv2d93(x287)
        return x288

m = M().eval()
x285 = torch.randn(torch.Size([1, 960, 1, 1]))
x281 = torch.randn(torch.Size([1, 960, 14, 14]))
start = time.time()
output = m(x285, x281)
end = time.time()
print(end-start)
