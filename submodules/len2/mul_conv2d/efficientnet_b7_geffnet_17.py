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
        self.conv2d86 = Conv2d(480, 80, kernel_size=(1, 1), stride=(1, 1), bias=False)

    def forward(self, x253, x258):
        x259=operator.mul(x253, x258)
        x260=self.conv2d86(x259)
        return x260

m = M().eval()
x253 = torch.randn(torch.Size([1, 480, 28, 28]))
x258 = torch.randn(torch.Size([1, 480, 1, 1]))
start = time.time()
output = m(x253, x258)
end = time.time()
print(end-start)
