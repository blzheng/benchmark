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
        self.conv2d141 = Conv2d(192, 48, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), bias=False)

    def forward(self, x501, x397, x404, x411, x418, x425, x432, x439, x446, x453, x460, x467, x474, x481, x488, x495, x509):
        x502=self.conv2d141(x501)
        x510=torch.cat([x397, x404, x411, x418, x425, x432, x439, x446, x453, x460, x467, x474, x481, x488, x495, x502, x509], 1)
        return x510

m = M().eval()
x501 = torch.randn(torch.Size([1, 192, 7, 7]))
x397 = torch.randn(torch.Size([1, 1056, 7, 7]))
x404 = torch.randn(torch.Size([1, 48, 7, 7]))
x411 = torch.randn(torch.Size([1, 48, 7, 7]))
x418 = torch.randn(torch.Size([1, 48, 7, 7]))
x425 = torch.randn(torch.Size([1, 48, 7, 7]))
x432 = torch.randn(torch.Size([1, 48, 7, 7]))
x439 = torch.randn(torch.Size([1, 48, 7, 7]))
x446 = torch.randn(torch.Size([1, 48, 7, 7]))
x453 = torch.randn(torch.Size([1, 48, 7, 7]))
x460 = torch.randn(torch.Size([1, 48, 7, 7]))
x467 = torch.randn(torch.Size([1, 48, 7, 7]))
x474 = torch.randn(torch.Size([1, 48, 7, 7]))
x481 = torch.randn(torch.Size([1, 48, 7, 7]))
x488 = torch.randn(torch.Size([1, 48, 7, 7]))
x495 = torch.randn(torch.Size([1, 48, 7, 7]))
x509 = torch.randn(torch.Size([1, 48, 7, 7]))
start = time.time()
output = m(x501, x397, x404, x411, x418, x425, x432, x439, x446, x453, x460, x467, x474, x481, x488, x495, x509)
end = time.time()
print(end-start)
