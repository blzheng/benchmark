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
        self.sigmoid9 = Sigmoid()
        self.conv2d82 = Conv2d(768, 192, kernel_size=(1, 1), stride=(1, 1), bias=False)

    def forward(self, x269, x265):
        x270=self.sigmoid9(x269)
        x271=operator.mul(x270, x265)
        x272=self.conv2d82(x271)
        return x272

m = M().eval()
x269 = torch.randn(torch.Size([1, 768, 1, 1]))
x265 = torch.randn(torch.Size([1, 768, 14, 14]))
start = time.time()
output = m(x269, x265)
end = time.time()
print(end-start)
