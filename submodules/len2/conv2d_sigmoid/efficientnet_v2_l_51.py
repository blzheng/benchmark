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
        self.conv2d291 = Conv2d(96, 2304, kernel_size=(1, 1), stride=(1, 1))
        self.sigmoid51 = Sigmoid()

    def forward(self, x936):
        x937=self.conv2d291(x936)
        x938=self.sigmoid51(x937)
        return x938

m = M().eval()
x936 = torch.randn(torch.Size([1, 96, 1, 1]))
start = time.time()
output = m(x936)
end = time.time()
print(end-start)
