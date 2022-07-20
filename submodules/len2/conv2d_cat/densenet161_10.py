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
        self.conv2d25 = Conv2d(192, 48, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), bias=False)

    def forward(self, x92, x51, x58, x65, x72, x79, x86, x100):
        x93=self.conv2d25(x92)
        x101=torch.cat([x51, x58, x65, x72, x79, x86, x93, x100], 1)
        return x101

m = M().eval()
x92 = torch.randn(torch.Size([1, 192, 28, 28]))
x51 = torch.randn(torch.Size([1, 192, 28, 28]))
x58 = torch.randn(torch.Size([1, 48, 28, 28]))
x65 = torch.randn(torch.Size([1, 48, 28, 28]))
x72 = torch.randn(torch.Size([1, 48, 28, 28]))
x79 = torch.randn(torch.Size([1, 48, 28, 28]))
x86 = torch.randn(torch.Size([1, 48, 28, 28]))
x100 = torch.randn(torch.Size([1, 48, 28, 28]))
start = time.time()
output = m(x92, x51, x58, x65, x72, x79, x86, x100)
end = time.time()
print(end-start)
