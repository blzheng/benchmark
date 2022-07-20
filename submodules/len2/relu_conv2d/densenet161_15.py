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
        self.relu16 = ReLU(inplace=True)
        self.conv2d16 = Conv2d(240, 192, kernel_size=(1, 1), stride=(1, 1), bias=False)

    def forward(self, x60):
        x61=self.relu16(x60)
        x62=self.conv2d16(x61)
        return x62

m = M().eval()
x60 = torch.randn(torch.Size([1, 240, 28, 28]))
start = time.time()
output = m(x60)
end = time.time()
print(end-start)
