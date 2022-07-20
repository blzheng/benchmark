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
        self.sigmoid6 = Sigmoid()
        self.conv2d53 = Conv2d(768, 160, kernel_size=(1, 1), stride=(1, 1), bias=False)

    def forward(self, x169, x165):
        x170=self.sigmoid6(x169)
        x171=operator.mul(x170, x165)
        x172=self.conv2d53(x171)
        return x172

m = M().eval()
x169 = torch.randn(torch.Size([1, 768, 1, 1]))
x165 = torch.randn(torch.Size([1, 768, 14, 14]))
start = time.time()
output = m(x169, x165)
end = time.time()
print(end-start)
