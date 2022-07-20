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
        self.conv2d11 = Conv2d(12, 104, kernel_size=(1, 1), stride=(1, 1))
        self.sigmoid1 = Sigmoid()

    def forward(self, x32):
        x33=self.conv2d11(x32)
        x34=self.sigmoid1(x33)
        return x34

m = M().eval()
x32 = torch.randn(torch.Size([1, 12, 1, 1]))
start = time.time()
output = m(x32)
end = time.time()
print(end-start)
