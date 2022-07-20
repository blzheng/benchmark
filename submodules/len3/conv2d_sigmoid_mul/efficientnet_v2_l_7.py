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
        self.conv2d71 = Conv2d(48, 768, kernel_size=(1, 1), stride=(1, 1))
        self.sigmoid7 = Sigmoid()

    def forward(self, x236, x233):
        x237=self.conv2d71(x236)
        x238=self.sigmoid7(x237)
        x239=operator.mul(x238, x233)
        return x239

m = M().eval()
x236 = torch.randn(torch.Size([1, 48, 1, 1]))
x233 = torch.randn(torch.Size([1, 768, 14, 14]))
start = time.time()
output = m(x236, x233)
end = time.time()
print(end-start)
