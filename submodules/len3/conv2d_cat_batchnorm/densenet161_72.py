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
        self.conv2d155 = Conv2d(192, 48, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), bias=False)
        self.batchnorm2d158 = BatchNorm2d(2160, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)

    def forward(self, x550, x397, x404, x411, x418, x425, x432, x439, x446, x453, x460, x467, x474, x481, x488, x495, x502, x509, x516, x523, x530, x537, x544, x558):
        x551=self.conv2d155(x550)
        x559=torch.cat([x397, x404, x411, x418, x425, x432, x439, x446, x453, x460, x467, x474, x481, x488, x495, x502, x509, x516, x523, x530, x537, x544, x551, x558], 1)
        x560=self.batchnorm2d158(x559)
        return x560

m = M().eval()
x550 = torch.randn(torch.Size([1, 192, 7, 7]))
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
x502 = torch.randn(torch.Size([1, 48, 7, 7]))
x509 = torch.randn(torch.Size([1, 48, 7, 7]))
x516 = torch.randn(torch.Size([1, 48, 7, 7]))
x523 = torch.randn(torch.Size([1, 48, 7, 7]))
x530 = torch.randn(torch.Size([1, 48, 7, 7]))
x537 = torch.randn(torch.Size([1, 48, 7, 7]))
x544 = torch.randn(torch.Size([1, 48, 7, 7]))
x558 = torch.randn(torch.Size([1, 48, 7, 7]))
start = time.time()
output = m(x550, x397, x404, x411, x418, x425, x432, x439, x446, x453, x460, x467, x474, x481, x488, x495, x502, x509, x516, x523, x530, x537, x544, x558)
end = time.time()
print(end-start)
