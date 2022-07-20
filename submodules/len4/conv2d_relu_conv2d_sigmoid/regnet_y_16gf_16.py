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
        self.conv2d86 = Conv2d(1232, 308, kernel_size=(1, 1), stride=(1, 1))
        self.relu67 = ReLU()
        self.conv2d87 = Conv2d(308, 1232, kernel_size=(1, 1), stride=(1, 1))
        self.sigmoid16 = Sigmoid()

    def forward(self, x272):
        x273=self.conv2d86(x272)
        x274=self.relu67(x273)
        x275=self.conv2d87(x274)
        x276=self.sigmoid16(x275)
        return x276

m = M().eval()
x272 = torch.randn(torch.Size([1, 1232, 1, 1]))
start = time.time()
output = m(x272)
end = time.time()
print(end-start)
