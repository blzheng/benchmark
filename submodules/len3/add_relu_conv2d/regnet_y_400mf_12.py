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
        self.relu52 = ReLU(inplace=True)
        self.conv2d70 = Conv2d(440, 440, kernel_size=(1, 1), stride=(1, 1), bias=False)

    def forward(self, x203, x217):
        x218=operator.add(x203, x217)
        x219=self.relu52(x218)
        x220=self.conv2d70(x219)
        return x220

m = M().eval()
x203 = torch.randn(torch.Size([1, 440, 7, 7]))
x217 = torch.randn(torch.Size([1, 440, 7, 7]))
start = time.time()
output = m(x203, x217)
end = time.time()
print(end-start)
