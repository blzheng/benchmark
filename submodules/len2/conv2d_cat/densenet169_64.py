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
        self.conv2d139 = Conv2d(128, 32, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), bias=False)

    def forward(self, x494, x369, x376, x383, x390, x397, x404, x411, x418, x425, x432, x439, x446, x453, x460, x467, x474, x481, x488, x502):
        x495=self.conv2d139(x494)
        x503=torch.cat([x369, x376, x383, x390, x397, x404, x411, x418, x425, x432, x439, x446, x453, x460, x467, x474, x481, x488, x495, x502], 1)
        return x503

m = M().eval()
x494 = torch.randn(torch.Size([1, 128, 7, 7]))
x369 = torch.randn(torch.Size([1, 640, 7, 7]))
x376 = torch.randn(torch.Size([1, 32, 7, 7]))
x383 = torch.randn(torch.Size([1, 32, 7, 7]))
x390 = torch.randn(torch.Size([1, 32, 7, 7]))
x397 = torch.randn(torch.Size([1, 32, 7, 7]))
x404 = torch.randn(torch.Size([1, 32, 7, 7]))
x411 = torch.randn(torch.Size([1, 32, 7, 7]))
x418 = torch.randn(torch.Size([1, 32, 7, 7]))
x425 = torch.randn(torch.Size([1, 32, 7, 7]))
x432 = torch.randn(torch.Size([1, 32, 7, 7]))
x439 = torch.randn(torch.Size([1, 32, 7, 7]))
x446 = torch.randn(torch.Size([1, 32, 7, 7]))
x453 = torch.randn(torch.Size([1, 32, 7, 7]))
x460 = torch.randn(torch.Size([1, 32, 7, 7]))
x467 = torch.randn(torch.Size([1, 32, 7, 7]))
x474 = torch.randn(torch.Size([1, 32, 7, 7]))
x481 = torch.randn(torch.Size([1, 32, 7, 7]))
x488 = torch.randn(torch.Size([1, 32, 7, 7]))
x502 = torch.randn(torch.Size([1, 32, 7, 7]))
start = time.time()
output = m(x494, x369, x376, x383, x390, x397, x404, x411, x418, x425, x432, x439, x446, x453, x460, x467, x474, x481, x488, x502)
end = time.time()
print(end-start)
