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
        self.sigmoid44 = Sigmoid()
        self.conv2d222 = Conv2d(3456, 576, kernel_size=(1, 1), stride=(1, 1), bias=False)

    def forward(self, x694, x690):
        x695=self.sigmoid44(x694)
        x696=operator.mul(x695, x690)
        x697=self.conv2d222(x696)
        return x697

m = M().eval()
x694 = torch.randn(torch.Size([1, 3456, 1, 1]))
x690 = torch.randn(torch.Size([1, 3456, 7, 7]))
start = time.time()
output = m(x694, x690)
end = time.time()
print(end-start)
