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
        self.batchnorm2d132 = BatchNorm2d(1088, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)

    def forward(self, x369, x376, x383, x390, x397, x404, x411, x418, x425, x432, x439, x446, x453, x460, x467):
        x468=torch.cat([x369, x376, x383, x390, x397, x404, x411, x418, x425, x432, x439, x446, x453, x460, x467], 1)
        x469=self.batchnorm2d132(x468)
        return x469

m = M().eval()
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
start = time.time()
output = m(x369, x376, x383, x390, x397, x404, x411, x418, x425, x432, x439, x446, x453, x460, x467)
end = time.time()
print(end-start)
