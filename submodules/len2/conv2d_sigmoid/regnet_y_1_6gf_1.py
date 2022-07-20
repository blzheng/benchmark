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
        self.conv2d10 = Conv2d(12, 48, kernel_size=(1, 1), stride=(1, 1))
        self.sigmoid1 = Sigmoid()

    def forward(self, x30):
        x31=self.conv2d10(x30)
        x32=self.sigmoid1(x31)
        return x32

m = M().eval()
x30 = torch.randn(torch.Size([1, 12, 1, 1]))
start = time.time()
output = m(x30)
end = time.time()
print(end-start)
