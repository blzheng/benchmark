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
        self.conv2d132 = Conv2d(726, 2904, kernel_size=(1, 1), stride=(1, 1))
        self.sigmoid25 = Sigmoid()

    def forward(self, x418, x415):
        x419=self.conv2d132(x418)
        x420=self.sigmoid25(x419)
        x421=operator.mul(x420, x415)
        return x421

m = M().eval()
x418 = torch.randn(torch.Size([1, 726, 1, 1]))
x415 = torch.randn(torch.Size([1, 2904, 14, 14]))
start = time.time()
output = m(x418, x415)
end = time.time()
print(end-start)
