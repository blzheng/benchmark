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

    def forward(self, x397, x404, x411, x418, x425, x432, x439, x446, x453, x460, x467):
        x468=torch.cat([x397, x404, x411, x418, x425, x432, x439, x446, x453, x460, x467], 1)
        return x468

m = M().eval()
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
start = time.time()
output = m(x397, x404, x411, x418, x425, x432, x439, x446, x453, x460, x467)
end = time.time()
print(end-start)
