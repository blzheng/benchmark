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
        self.batchnorm2d138 = BatchNorm2d(1680, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)

    def forward(self, x397, x404, x411, x418, x425, x432, x439, x446, x453, x460, x467, x474, x481, x488):
        x489=torch.cat([x397, x404, x411, x418, x425, x432, x439, x446, x453, x460, x467, x474, x481, x488], 1)
        x490=self.batchnorm2d138(x489)
        return x490

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
x474 = torch.randn(torch.Size([1, 48, 7, 7]))
x481 = torch.randn(torch.Size([1, 48, 7, 7]))
x488 = torch.randn(torch.Size([1, 48, 7, 7]))
start = time.time()
output = m(x397, x404, x411, x418, x425, x432, x439, x446, x453, x460, x467, x474, x481, x488)
end = time.time()
print(end-start)
