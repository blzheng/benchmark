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
        self.conv2d181 = Conv2d(76, 1824, kernel_size=(1, 1), stride=(1, 1))
        self.sigmoid36 = Sigmoid()

    def forward(self, x567):
        x568=self.conv2d181(x567)
        x569=self.sigmoid36(x568)
        return x569

m = M().eval()
x567 = torch.randn(torch.Size([1, 76, 1, 1]))
start = time.time()
output = m(x567)
end = time.time()
print(end-start)
