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
        self.conv2d37 = Conv2d(128, 128, kernel_size=(7, 1), stride=(1, 1), padding=(3, 0), bias=False)

    def forward(self, x133):
        x134=torch.nn.functional.relu(x133,inplace=True)
        x135=self.conv2d37(x134)
        return x135

m = M().eval()
x133 = torch.randn(torch.Size([1, 128, 12, 12]))
start = time.time()
output = m(x133)
end = time.time()
print(end-start)
