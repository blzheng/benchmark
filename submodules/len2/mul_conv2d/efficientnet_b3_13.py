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
        self.conv2d68 = Conv2d(576, 136, kernel_size=(1, 1), stride=(1, 1), bias=False)

    def forward(self, x208, x203):
        x209=operator.mul(x208, x203)
        x210=self.conv2d68(x209)
        return x210

m = M().eval()
x208 = torch.randn(torch.Size([1, 576, 1, 1]))
x203 = torch.randn(torch.Size([1, 576, 14, 14]))
start = time.time()
output = m(x208, x203)
end = time.time()
print(end-start)