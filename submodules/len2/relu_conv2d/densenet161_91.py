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
        self.relu92 = ReLU(inplace=True)
        self.conv2d92 = Conv2d(192, 48, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), bias=False)

    def forward(self, x327):
        x328=self.relu92(x327)
        x329=self.conv2d92(x328)
        return x329

m = M().eval()
x327 = torch.randn(torch.Size([1, 192, 14, 14]))
start = time.time()
output = m(x327)
end = time.time()
print(end-start)
