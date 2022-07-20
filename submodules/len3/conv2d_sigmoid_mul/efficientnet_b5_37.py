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
        self.conv2d186 = Conv2d(128, 3072, kernel_size=(1, 1), stride=(1, 1))
        self.sigmoid37 = Sigmoid()

    def forward(self, x581, x578):
        x582=self.conv2d186(x581)
        x583=self.sigmoid37(x582)
        x584=operator.mul(x583, x578)
        return x584

m = M().eval()
x581 = torch.randn(torch.Size([1, 128, 1, 1]))
x578 = torch.randn(torch.Size([1, 3072, 7, 7]))
start = time.time()
output = m(x581, x578)
end = time.time()
print(end-start)
