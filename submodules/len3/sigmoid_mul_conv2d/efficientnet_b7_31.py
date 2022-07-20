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
        self.sigmoid31 = Sigmoid()
        self.conv2d156 = Conv2d(1344, 224, kernel_size=(1, 1), stride=(1, 1), bias=False)

    def forward(self, x487, x483):
        x488=self.sigmoid31(x487)
        x489=operator.mul(x488, x483)
        x490=self.conv2d156(x489)
        return x490

m = M().eval()
x487 = torch.randn(torch.Size([1, 1344, 1, 1]))
x483 = torch.randn(torch.Size([1, 1344, 14, 14]))
start = time.time()
output = m(x487, x483)
end = time.time()
print(end-start)
