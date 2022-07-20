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
        self.relu61 = ReLU(inplace=True)
        self.conv2d65 = Conv2d(256, 256, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), bias=False)

    def forward(self, x212):
        x213=self.relu61(x212)
        x214=self.conv2d65(x213)
        return x214

m = M().eval()
x212 = torch.randn(torch.Size([1, 256, 14, 14]))
start = time.time()
output = m(x212)
end = time.time()
print(end-start)
