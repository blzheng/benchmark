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
        self.conv2d99 = Conv2d(128, 32, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), bias=False)

    def forward(self, x354, x313, x320, x327, x334, x341, x348, x362):
        x355=self.conv2d99(x354)
        x363=torch.cat([x313, x320, x327, x334, x341, x348, x355, x362], 1)
        return x363

m = M().eval()
x354 = torch.randn(torch.Size([1, 128, 7, 7]))
x313 = torch.randn(torch.Size([1, 512, 7, 7]))
x320 = torch.randn(torch.Size([1, 32, 7, 7]))
x327 = torch.randn(torch.Size([1, 32, 7, 7]))
x334 = torch.randn(torch.Size([1, 32, 7, 7]))
x341 = torch.randn(torch.Size([1, 32, 7, 7]))
x348 = torch.randn(torch.Size([1, 32, 7, 7]))
x362 = torch.randn(torch.Size([1, 32, 7, 7]))
start = time.time()
output = m(x354, x313, x320, x327, x334, x341, x348, x362)
end = time.time()
print(end-start)