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
        self.conv2d137 = Conv2d(128, 32, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), bias=False)
        self.batchnorm2d140 = BatchNorm2d(1216, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)

    def forward(self, x487, x369, x376, x383, x390, x397, x404, x411, x418, x425, x432, x439, x446, x453, x460, x467, x474, x481, x495):
        x488=self.conv2d137(x487)
        x496=torch.cat([x369, x376, x383, x390, x397, x404, x411, x418, x425, x432, x439, x446, x453, x460, x467, x474, x481, x488, x495], 1)
        x497=self.batchnorm2d140(x496)
        return x497

m = M().eval()
x487 = torch.randn(torch.Size([1, 128, 7, 7]))
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
x495 = torch.randn(torch.Size([1, 32, 7, 7]))
start = time.time()
output = m(x487, x369, x376, x383, x390, x397, x404, x411, x418, x425, x432, x439, x446, x453, x460, x467, x474, x481, x495)
end = time.time()
print(end-start)
