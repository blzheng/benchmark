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
        self.conv2d261 = Conv2d(96, 2304, kernel_size=(1, 1), stride=(1, 1))
        self.sigmoid45 = Sigmoid()

    def forward(self, x840):
        x841=self.conv2d261(x840)
        x842=self.sigmoid45(x841)
        return x842

m = M().eval()
x840 = torch.randn(torch.Size([1, 96, 1, 1]))
start = time.time()
output = m(x840)
end = time.time()
print(end-start)
