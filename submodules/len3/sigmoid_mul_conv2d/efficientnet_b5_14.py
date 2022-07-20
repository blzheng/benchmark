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
        self.sigmoid14 = Sigmoid()
        self.conv2d72 = Conv2d(768, 128, kernel_size=(1, 1), stride=(1, 1), bias=False)

    def forward(self, x220, x216):
        x221=self.sigmoid14(x220)
        x222=operator.mul(x221, x216)
        x223=self.conv2d72(x222)
        return x223

m = M().eval()
x220 = torch.randn(torch.Size([1, 768, 1, 1]))
x216 = torch.randn(torch.Size([1, 768, 14, 14]))
start = time.time()
output = m(x220, x216)
end = time.time()
print(end-start)