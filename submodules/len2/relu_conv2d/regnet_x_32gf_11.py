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
        self.relu11 = ReLU(inplace=True)
        self.conv2d14 = Conv2d(672, 672, kernel_size=(1, 1), stride=(1, 1), bias=False)

    def forward(self, x42):
        x43=self.relu11(x42)
        x44=self.conv2d14(x43)
        return x44

m = M().eval()
x42 = torch.randn(torch.Size([1, 672, 28, 28]))
start = time.time()
output = m(x42)
end = time.time()
print(end-start)
