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
        self.relu14 = ReLU(inplace=True)
        self.conv2d18 = Conv2d(288, 288, kernel_size=(1, 1), stride=(1, 1), bias=False)

    def forward(self, x54):
        x55=self.relu14(x54)
        x56=self.conv2d18(x55)
        return x56

m = M().eval()
x54 = torch.randn(torch.Size([1, 288, 14, 14]))
start = time.time()
output = m(x54)
end = time.time()
print(end-start)
