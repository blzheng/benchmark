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
        self.sigmoid35 = Sigmoid()
        self.conv2d177 = Conv2d(1824, 304, kernel_size=(1, 1), stride=(1, 1), bias=False)

    def forward(self, x552, x548):
        x553=self.sigmoid35(x552)
        x554=operator.mul(x553, x548)
        x555=self.conv2d177(x554)
        return x555

m = M().eval()
x552 = torch.randn(torch.Size([1, 1824, 1, 1]))
x548 = torch.randn(torch.Size([1, 1824, 7, 7]))
start = time.time()
output = m(x552, x548)
end = time.time()
print(end-start)
