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

    def forward(self, x487, x481, x495):
        x488=self.conv2d137(x487)
        x496=torch.cat([x481, x488, x495], 1)
        return x496

m = M().eval()
x487 = torch.randn(torch.Size([1, 128, 7, 7]))
x481 = torch.randn(torch.Size([1, 896, 7, 7]))
x495 = torch.randn(torch.Size([1, 32, 7, 7]))
start = time.time()
output = m(x487, x481, x495)
end = time.time()
print(end-start)
