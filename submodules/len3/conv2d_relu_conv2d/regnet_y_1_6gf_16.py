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
        self.conv2d86 = Conv2d(336, 84, kernel_size=(1, 1), stride=(1, 1))
        self.relu67 = ReLU()
        self.conv2d87 = Conv2d(84, 336, kernel_size=(1, 1), stride=(1, 1))

    def forward(self, x272):
        x273=self.conv2d86(x272)
        x274=self.relu67(x273)
        x275=self.conv2d87(x274)
        return x275

m = M().eval()
x272 = torch.randn(torch.Size([1, 336, 1, 1]))
start = time.time()
output = m(x272)
end = time.time()
print(end-start)
