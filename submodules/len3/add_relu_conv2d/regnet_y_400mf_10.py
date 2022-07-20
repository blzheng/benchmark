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
        self.relu44 = ReLU(inplace=True)
        self.conv2d60 = Conv2d(440, 440, kernel_size=(1, 1), stride=(1, 1), bias=False)

    def forward(self, x171, x185):
        x186=operator.add(x171, x185)
        x187=self.relu44(x186)
        x188=self.conv2d60(x187)
        return x188

m = M().eval()
x171 = torch.randn(torch.Size([1, 440, 7, 7]))
x185 = torch.randn(torch.Size([1, 440, 7, 7]))
start = time.time()
output = m(x171, x185)
end = time.time()
print(end-start)
