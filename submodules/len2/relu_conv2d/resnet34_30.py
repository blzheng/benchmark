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
        self.relu31 = ReLU(inplace=True)
        self.conv2d35 = Conv2d(512, 512, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), bias=False)

    def forward(self, x117):
        x118=self.relu31(x117)
        x119=self.conv2d35(x118)
        return x119

m = M().eval()
x117 = torch.randn(torch.Size([1, 512, 7, 7]))
start = time.time()
output = m(x117)
end = time.time()
print(end-start)
