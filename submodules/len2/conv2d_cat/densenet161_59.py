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
        self.conv2d129 = Conv2d(192, 48, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), bias=False)

    def forward(self, x459, x397, x404, x411, x418, x425, x432, x439, x446, x453, x467):
        x460=self.conv2d129(x459)
        x468=torch.cat([x397, x404, x411, x418, x425, x432, x439, x446, x453, x460, x467], 1)
        return x468

m = M().eval()
x459 = torch.randn(torch.Size([1, 192, 7, 7]))
x397 = torch.randn(torch.Size([1, 1056, 7, 7]))
x404 = torch.randn(torch.Size([1, 48, 7, 7]))
x411 = torch.randn(torch.Size([1, 48, 7, 7]))
x418 = torch.randn(torch.Size([1, 48, 7, 7]))
x425 = torch.randn(torch.Size([1, 48, 7, 7]))
x432 = torch.randn(torch.Size([1, 48, 7, 7]))
x439 = torch.randn(torch.Size([1, 48, 7, 7]))
x446 = torch.randn(torch.Size([1, 48, 7, 7]))
x453 = torch.randn(torch.Size([1, 48, 7, 7]))
x467 = torch.randn(torch.Size([1, 48, 7, 7]))
start = time.time()
output = m(x459, x397, x404, x411, x418, x425, x432, x439, x446, x453, x467)
end = time.time()
print(end-start)
