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
        self.conv2d70 = Conv2d(40, 10, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), bias=False)

    def forward(self, x42):
        x213=self.conv2d70(x42)
        return x213

m = M().eval()
x42 = torch.randn(torch.Size([1, 40, 28, 28]))
start = time.time()
output = m(x42)
end = time.time()
print(end-start)
