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
        self.batchnorm2d160 = BatchNorm2d(1280, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)

    def forward(self, x481, x488, x495, x502, x509, x516, x523, x530, x537, x544, x551, x558, x565):
        x566=torch.cat([x481, x488, x495, x502, x509, x516, x523, x530, x537, x544, x551, x558, x565], 1)
        x567=self.batchnorm2d160(x566)
        return x567

m = M().eval()
x481 = torch.randn(torch.Size([1, 896, 7, 7]))
x488 = torch.randn(torch.Size([1, 32, 7, 7]))
x495 = torch.randn(torch.Size([1, 32, 7, 7]))
x502 = torch.randn(torch.Size([1, 32, 7, 7]))
x509 = torch.randn(torch.Size([1, 32, 7, 7]))
x516 = torch.randn(torch.Size([1, 32, 7, 7]))
x523 = torch.randn(torch.Size([1, 32, 7, 7]))
x530 = torch.randn(torch.Size([1, 32, 7, 7]))
x537 = torch.randn(torch.Size([1, 32, 7, 7]))
x544 = torch.randn(torch.Size([1, 32, 7, 7]))
x551 = torch.randn(torch.Size([1, 32, 7, 7]))
x558 = torch.randn(torch.Size([1, 32, 7, 7]))
x565 = torch.randn(torch.Size([1, 32, 7, 7]))
start = time.time()
output = m(x481, x488, x495, x502, x509, x516, x523, x530, x537, x544, x551, x558, x565)
end = time.time()
print(end-start)
