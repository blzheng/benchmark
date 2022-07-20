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
        self.conv2d21 = Conv2d(10, 240, kernel_size=(1, 1), stride=(1, 1))
        self.sigmoid4 = Sigmoid()

    def forward(self, x63):
        x64=self.conv2d21(x63)
        x65=self.sigmoid4(x64)
        return x65

m = M().eval()
x63 = torch.randn(torch.Size([1, 10, 1, 1]))
start = time.time()
output = m(x63)
end = time.time()
print(end-start)