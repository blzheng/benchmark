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
        self.conv2d185 = Conv2d(56, 1344, kernel_size=(1, 1), stride=(1, 1))
        self.sigmoid37 = Sigmoid()

    def forward(self, x582, x579):
        x583=self.conv2d185(x582)
        x584=self.sigmoid37(x583)
        x585=operator.mul(x584, x579)
        return x585

m = M().eval()
x582 = torch.randn(torch.Size([1, 56, 1, 1]))
x579 = torch.randn(torch.Size([1, 1344, 14, 14]))
start = time.time()
output = m(x582, x579)
end = time.time()
print(end-start)
