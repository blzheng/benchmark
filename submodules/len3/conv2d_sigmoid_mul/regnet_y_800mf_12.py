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
        self.conv2d68 = Conv2d(80, 784, kernel_size=(1, 1), stride=(1, 1))
        self.sigmoid12 = Sigmoid()

    def forward(self, x212, x209):
        x213=self.conv2d68(x212)
        x214=self.sigmoid12(x213)
        x215=operator.mul(x214, x209)
        return x215

m = M().eval()
x212 = torch.randn(torch.Size([1, 80, 1, 1]))
x209 = torch.randn(torch.Size([1, 784, 7, 7]))
start = time.time()
output = m(x212, x209)
end = time.time()
print(end-start)
