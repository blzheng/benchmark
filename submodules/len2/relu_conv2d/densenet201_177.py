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
        self.relu178 = ReLU(inplace=True)
        self.conv2d178 = Conv2d(1568, 128, kernel_size=(1, 1), stride=(1, 1), bias=False)

    def forward(self, x630):
        x631=self.relu178(x630)
        x632=self.conv2d178(x631)
        return x632

m = M().eval()
x630 = torch.randn(torch.Size([1, 1568, 7, 7]))
start = time.time()
output = m(x630)
end = time.time()
print(end-start)
