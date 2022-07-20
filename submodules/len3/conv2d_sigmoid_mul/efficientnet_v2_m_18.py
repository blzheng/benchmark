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
        self.conv2d117 = Conv2d(44, 1056, kernel_size=(1, 1), stride=(1, 1))
        self.sigmoid18 = Sigmoid()

    def forward(self, x377, x374):
        x378=self.conv2d117(x377)
        x379=self.sigmoid18(x378)
        x380=operator.mul(x379, x374)
        return x380

m = M().eval()
x377 = torch.randn(torch.Size([1, 44, 1, 1]))
x374 = torch.randn(torch.Size([1, 1056, 14, 14]))
start = time.time()
output = m(x377, x374)
end = time.time()
print(end-start)
