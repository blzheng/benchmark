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
        self.conv2d69 = Conv2d(176, 1056, kernel_size=(1, 1), stride=(1, 1), bias=False)

    def forward(self, x223, x208):
        x224=operator.add(x223, x208)
        x225=self.conv2d69(x224)
        return x225

m = M().eval()
x223 = torch.randn(torch.Size([1, 176, 14, 14]))
x208 = torch.randn(torch.Size([1, 176, 14, 14]))
start = time.time()
output = m(x223, x208)
end = time.time()
print(end-start)
