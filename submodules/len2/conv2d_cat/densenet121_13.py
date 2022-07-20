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
        self.conv2d31 = Conv2d(128, 32, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), bias=False)

    def forward(self, x113, x51, x58, x65, x72, x79, x86, x93, x100, x107, x121):
        x114=self.conv2d31(x113)
        x122=torch.cat([x51, x58, x65, x72, x79, x86, x93, x100, x107, x114, x121], 1)
        return x122

m = M().eval()
x113 = torch.randn(torch.Size([1, 128, 28, 28]))
x51 = torch.randn(torch.Size([1, 128, 28, 28]))
x58 = torch.randn(torch.Size([1, 32, 28, 28]))
x65 = torch.randn(torch.Size([1, 32, 28, 28]))
x72 = torch.randn(torch.Size([1, 32, 28, 28]))
x79 = torch.randn(torch.Size([1, 32, 28, 28]))
x86 = torch.randn(torch.Size([1, 32, 28, 28]))
x93 = torch.randn(torch.Size([1, 32, 28, 28]))
x100 = torch.randn(torch.Size([1, 32, 28, 28]))
x107 = torch.randn(torch.Size([1, 32, 28, 28]))
x121 = torch.randn(torch.Size([1, 32, 28, 28]))
start = time.time()
output = m(x113, x51, x58, x65, x72, x79, x86, x93, x100, x107, x121)
end = time.time()
print(end-start)
