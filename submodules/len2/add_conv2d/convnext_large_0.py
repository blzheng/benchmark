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
        self.conv2d2 = Conv2d(192, 192, kernel_size=(7, 7), stride=(1, 1), padding=(3, 3), groups=192)

    def forward(self, x16, x6):
        x17=operator.add(x16, x6)
        x19=self.conv2d2(x17)
        return x19

m = M().eval()
x16 = torch.randn(torch.Size([1, 192, 56, 56]))
x6 = torch.randn(torch.Size([1, 192, 56, 56]))
start = time.time()
output = m(x16, x6)
end = time.time()
print(end-start)
