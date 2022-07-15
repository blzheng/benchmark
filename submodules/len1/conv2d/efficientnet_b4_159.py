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
        self.conv2d159 = Conv2d(448, 1792, kernel_size=(1, 1), stride=(1, 1), bias=False)

    def forward(self, x495):
        x496=self.conv2d159(x495)
        return x496

m = M().eval()
x495 = torch.randn(torch.Size([1, 448, 7, 7]))
start = time.time()
output = m(x495)
end = time.time()
print(end-start)
