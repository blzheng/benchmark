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
        self.conv2d17 = Conv2d(96, 96, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), bias=False)

    def forward(self, x67):
        x68=torch.nn.functional.relu(x67,inplace=True)
        x69=self.conv2d17(x68)
        return x69

m = M().eval()
x67 = torch.randn(torch.Size([1, 96, 25, 25]))
start = time.time()
output = m(x67)
end = time.time()
print(end-start)
