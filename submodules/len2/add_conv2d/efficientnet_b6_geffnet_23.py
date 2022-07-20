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
        self.conv2d143 = Conv2d(200, 1200, kernel_size=(1, 1), stride=(1, 1), bias=False)

    def forward(self, x426, x412):
        x427=operator.add(x426, x412)
        x428=self.conv2d143(x427)
        return x428

m = M().eval()
x426 = torch.randn(torch.Size([1, 200, 14, 14]))
x412 = torch.randn(torch.Size([1, 200, 14, 14]))
start = time.time()
output = m(x426, x412)
end = time.time()
print(end-start)
