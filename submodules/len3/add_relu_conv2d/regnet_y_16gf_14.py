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
        self.relu60 = ReLU(inplace=True)
        self.conv2d79 = Conv2d(1232, 1232, kernel_size=(1, 1), stride=(1, 1), bias=False)

    def forward(self, x233, x247):
        x248=operator.add(x233, x247)
        x249=self.relu60(x248)
        x250=self.conv2d79(x249)
        return x250

m = M().eval()
x233 = torch.randn(torch.Size([1, 1232, 14, 14]))
x247 = torch.randn(torch.Size([1, 1232, 14, 14]))
start = time.time()
output = m(x233, x247)
end = time.time()
print(end-start)
