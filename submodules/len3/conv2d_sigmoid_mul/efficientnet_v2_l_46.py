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
        self.conv2d266 = Conv2d(96, 2304, kernel_size=(1, 1), stride=(1, 1))
        self.sigmoid46 = Sigmoid()

    def forward(self, x856, x853):
        x857=self.conv2d266(x856)
        x858=self.sigmoid46(x857)
        x859=operator.mul(x858, x853)
        return x859

m = M().eval()
x856 = torch.randn(torch.Size([1, 96, 1, 1]))
x853 = torch.randn(torch.Size([1, 2304, 7, 7]))
start = time.time()
output = m(x856, x853)
end = time.time()
print(end-start)
