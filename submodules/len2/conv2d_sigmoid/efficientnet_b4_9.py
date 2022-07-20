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
        self.conv2d47 = Conv2d(14, 336, kernel_size=(1, 1), stride=(1, 1))
        self.sigmoid9 = Sigmoid()

    def forward(self, x144):
        x145=self.conv2d47(x144)
        x146=self.sigmoid9(x145)
        return x146

m = M().eval()
x144 = torch.randn(torch.Size([1, 14, 1, 1]))
start = time.time()
output = m(x144)
end = time.time()
print(end-start)
