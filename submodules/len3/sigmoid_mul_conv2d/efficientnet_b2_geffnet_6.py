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
        self.conv2d33 = Conv2d(288, 48, kernel_size=(1, 1), stride=(1, 1), bias=False)

    def forward(self, x96, x92):
        x97=x96.sigmoid()
        x98=operator.mul(x92, x97)
        x99=self.conv2d33(x98)
        return x99

m = M().eval()
x96 = torch.randn(torch.Size([1, 288, 1, 1]))
x92 = torch.randn(torch.Size([1, 288, 28, 28]))
start = time.time()
output = m(x96, x92)
end = time.time()
print(end-start)
