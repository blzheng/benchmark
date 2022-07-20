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
        self.relu134 = ReLU(inplace=True)
        self.conv2d134 = Conv2d(128, 32, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), bias=False)

    def forward(self, x474):
        x475=self.relu134(x474)
        x476=self.conv2d134(x475)
        return x476

m = M().eval()
x474 = torch.randn(torch.Size([1, 128, 14, 14]))
start = time.time()
output = m(x474)
end = time.time()
print(end-start)
