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
        self.conv2d149 = Conv2d(128, 32, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), bias=False)
        self.batchnorm2d152 = BatchNorm2d(1408, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)

    def forward(self, x529, x369, x376, x383, x390, x397, x404, x411, x418, x425, x432, x439, x446, x453, x460, x467, x474, x481, x488, x495, x502, x509, x516, x523, x537):
        x530=self.conv2d149(x529)
        x538=torch.cat([x369, x376, x383, x390, x397, x404, x411, x418, x425, x432, x439, x446, x453, x460, x467, x474, x481, x488, x495, x502, x509, x516, x523, x530, x537], 1)
        x539=self.batchnorm2d152(x538)
        return x539

m = M().eval()
x529 = torch.randn(torch.Size([1, 128, 7, 7]))
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
x495 = torch.randn(torch.Size([1, 32, 7, 7]))
x502 = torch.randn(torch.Size([1, 32, 7, 7]))
x509 = torch.randn(torch.Size([1, 32, 7, 7]))
x516 = torch.randn(torch.Size([1, 32, 7, 7]))
x523 = torch.randn(torch.Size([1, 32, 7, 7]))
x537 = torch.randn(torch.Size([1, 32, 7, 7]))
start = time.time()
output = m(x529, x369, x376, x383, x390, x397, x404, x411, x418, x425, x432, x439, x446, x453, x460, x467, x474, x481, x488, x495, x502, x509, x516, x523, x537)
end = time.time()
print(end-start)
