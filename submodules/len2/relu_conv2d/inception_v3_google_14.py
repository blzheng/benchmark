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
        self.conv2d32 = Conv2d(128, 128, kernel_size=(1, 7), stride=(1, 1), padding=(0, 3), bias=False)

    def forward(self, x118):
        x119=torch.nn.functional.relu(x118,inplace=True)
        x120=self.conv2d32(x119)
        return x120

m = M().eval()
x118 = torch.randn(torch.Size([1, 128, 12, 12]))
start = time.time()
output = m(x118)
end = time.time()
print(end-start)