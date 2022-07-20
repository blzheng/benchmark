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
        self.conv2d35 = Conv2d(192, 48, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), bias=False)
        self.batchnorm2d38 = BatchNorm2d(768, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)

    def forward(self, x127, x51, x58, x65, x72, x79, x86, x93, x100, x107, x114, x121, x135):
        x128=self.conv2d35(x127)
        x136=torch.cat([x51, x58, x65, x72, x79, x86, x93, x100, x107, x114, x121, x128, x135], 1)
        x137=self.batchnorm2d38(x136)
        return x137

m = M().eval()
x127 = torch.randn(torch.Size([1, 192, 28, 28]))
x51 = torch.randn(torch.Size([1, 192, 28, 28]))
x58 = torch.randn(torch.Size([1, 48, 28, 28]))
x65 = torch.randn(torch.Size([1, 48, 28, 28]))
x72 = torch.randn(torch.Size([1, 48, 28, 28]))
x79 = torch.randn(torch.Size([1, 48, 28, 28]))
x86 = torch.randn(torch.Size([1, 48, 28, 28]))
x93 = torch.randn(torch.Size([1, 48, 28, 28]))
x100 = torch.randn(torch.Size([1, 48, 28, 28]))
x107 = torch.randn(torch.Size([1, 48, 28, 28]))
x114 = torch.randn(torch.Size([1, 48, 28, 28]))
x121 = torch.randn(torch.Size([1, 48, 28, 28]))
x135 = torch.randn(torch.Size([1, 48, 28, 28]))
start = time.time()
output = m(x127, x51, x58, x65, x72, x79, x86, x93, x100, x107, x114, x121, x135)
end = time.time()
print(end-start)
