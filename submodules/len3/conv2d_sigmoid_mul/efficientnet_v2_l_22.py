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
        self.conv2d146 = Conv2d(56, 1344, kernel_size=(1, 1), stride=(1, 1))
        self.sigmoid22 = Sigmoid()

    def forward(self, x474, x471):
        x475=self.conv2d146(x474)
        x476=self.sigmoid22(x475)
        x477=operator.mul(x476, x471)
        return x477

m = M().eval()
x474 = torch.randn(torch.Size([1, 56, 1, 1]))
x471 = torch.randn(torch.Size([1, 1344, 14, 14]))
start = time.time()
output = m(x474, x471)
end = time.time()
print(end-start)
