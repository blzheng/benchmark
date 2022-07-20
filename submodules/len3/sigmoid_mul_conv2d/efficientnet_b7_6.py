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
        self.sigmoid6 = Sigmoid()
        self.conv2d31 = Conv2d(288, 48, kernel_size=(1, 1), stride=(1, 1), bias=False)

    def forward(self, x93, x89):
        x94=self.sigmoid6(x93)
        x95=operator.mul(x94, x89)
        x96=self.conv2d31(x95)
        return x96

m = M().eval()
x93 = torch.randn(torch.Size([1, 288, 1, 1]))
x89 = torch.randn(torch.Size([1, 288, 56, 56]))
start = time.time()
output = m(x93, x89)
end = time.time()
print(end-start)
