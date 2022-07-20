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
        self.conv2d135 = Conv2d(40, 960, kernel_size=(1, 1), stride=(1, 1))
        self.sigmoid27 = Sigmoid()
        self.conv2d136 = Conv2d(960, 160, kernel_size=(1, 1), stride=(1, 1), bias=False)

    def forward(self, x424, x421):
        x425=self.conv2d135(x424)
        x426=self.sigmoid27(x425)
        x427=operator.mul(x426, x421)
        x428=self.conv2d136(x427)
        return x428

m = M().eval()
x424 = torch.randn(torch.Size([1, 40, 1, 1]))
x421 = torch.randn(torch.Size([1, 960, 14, 14]))
start = time.time()
output = m(x424, x421)
end = time.time()
print(end-start)
