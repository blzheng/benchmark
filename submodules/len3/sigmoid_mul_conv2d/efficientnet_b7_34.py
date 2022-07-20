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
        self.sigmoid34 = Sigmoid()
        self.conv2d171 = Conv2d(1344, 224, kernel_size=(1, 1), stride=(1, 1), bias=False)

    def forward(self, x535, x531):
        x536=self.sigmoid34(x535)
        x537=operator.mul(x536, x531)
        x538=self.conv2d171(x537)
        return x538

m = M().eval()
x535 = torch.randn(torch.Size([1, 1344, 1, 1]))
x531 = torch.randn(torch.Size([1, 1344, 14, 14]))
start = time.time()
output = m(x535, x531)
end = time.time()
print(end-start)
