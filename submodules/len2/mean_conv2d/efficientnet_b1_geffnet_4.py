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
        self.conv2d21 = Conv2d(144, 6, kernel_size=(1, 1), stride=(1, 1))

    def forward(self, x63):
        x64=x63.mean((2, 3),keepdim=True)
        x65=self.conv2d21(x64)
        return x65

m = M().eval()
x63 = torch.randn(torch.Size([1, 144, 56, 56]))
start = time.time()
output = m(x63)
end = time.time()
print(end-start)
