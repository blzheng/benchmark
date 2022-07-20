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
        self.conv2d53 = Conv2d(192, 768, kernel_size=(1, 1), stride=(1, 1), bias=False)

    def forward(self, x178, x163):
        x179=operator.add(x178, x163)
        x180=self.conv2d53(x179)
        return x180

m = M().eval()
x178 = torch.randn(torch.Size([1, 192, 14, 14]))
x163 = torch.randn(torch.Size([1, 192, 14, 14]))
start = time.time()
output = m(x178, x163)
end = time.time()
print(end-start)