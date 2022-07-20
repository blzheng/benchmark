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
        self.conv2d186 = Conv2d(1344, 224, kernel_size=(1, 1), stride=(1, 1), bias=False)

    def forward(self, x584, x579):
        x585=operator.mul(x584, x579)
        x586=self.conv2d186(x585)
        return x586

m = M().eval()
x584 = torch.randn(torch.Size([1, 1344, 1, 1]))
x579 = torch.randn(torch.Size([1, 1344, 14, 14]))
start = time.time()
output = m(x584, x579)
end = time.time()
print(end-start)
