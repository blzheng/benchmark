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
        self.relu19 = ReLU(inplace=True)
        self.conv2d29 = Conv2d(480, 80, kernel_size=(1, 1), stride=(1, 1), bias=False)

    def forward(self, x82):
        x83=self.relu19(x82)
        x84=self.conv2d29(x83)
        return x84

m = M().eval()
x82 = torch.randn(torch.Size([1, 480, 14, 14]))
start = time.time()
output = m(x82)
end = time.time()
print(end-start)
