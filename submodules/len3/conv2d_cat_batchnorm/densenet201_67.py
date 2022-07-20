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
        self.conv2d145 = Conv2d(128, 32, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), bias=False)
        self.batchnorm2d148 = BatchNorm2d(1088, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)

    def forward(self, x515, x481, x488, x495, x502, x509, x523):
        x516=self.conv2d145(x515)
        x524=torch.cat([x481, x488, x495, x502, x509, x516, x523], 1)
        x525=self.batchnorm2d148(x524)
        return x525

m = M().eval()
x515 = torch.randn(torch.Size([1, 128, 7, 7]))
x481 = torch.randn(torch.Size([1, 896, 7, 7]))
x488 = torch.randn(torch.Size([1, 32, 7, 7]))
x495 = torch.randn(torch.Size([1, 32, 7, 7]))
x502 = torch.randn(torch.Size([1, 32, 7, 7]))
x509 = torch.randn(torch.Size([1, 32, 7, 7]))
x523 = torch.randn(torch.Size([1, 32, 7, 7]))
start = time.time()
output = m(x515, x481, x488, x495, x502, x509, x523)
end = time.time()
print(end-start)
