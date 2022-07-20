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
        self.conv2d176 = Conv2d(56, 1344, kernel_size=(1, 1), stride=(1, 1))
        self.sigmoid28 = Sigmoid()

    def forward(self, x570, x567):
        x571=self.conv2d176(x570)
        x572=self.sigmoid28(x571)
        x573=operator.mul(x572, x567)
        return x573

m = M().eval()
x570 = torch.randn(torch.Size([1, 56, 1, 1]))
x567 = torch.randn(torch.Size([1, 1344, 14, 14]))
start = time.time()
output = m(x570, x567)
end = time.time()
print(end-start)
