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
        self.relu9 = ReLU(inplace=True)
        self.conv2d12 = Conv2d(128, 128, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), bias=False)

    def forward(self, x39, x34):
        x40=operator.add(x39, x34)
        x41=self.relu9(x40)
        x42=self.conv2d12(x41)
        return x42

m = M().eval()
x39 = torch.randn(torch.Size([1, 128, 28, 28]))
x34 = torch.randn(torch.Size([1, 128, 28, 28]))
start = time.time()
output = m(x39, x34)
end = time.time()
print(end-start)
