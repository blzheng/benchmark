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
        self.sigmoid20 = Sigmoid()
        self.conv2d128 = Conv2d(1056, 176, kernel_size=(1, 1), stride=(1, 1), bias=False)

    def forward(self, x410, x406):
        x411=self.sigmoid20(x410)
        x412=operator.mul(x411, x406)
        x413=self.conv2d128(x412)
        return x413

m = M().eval()
x410 = torch.randn(torch.Size([1, 1056, 1, 1]))
x406 = torch.randn(torch.Size([1, 1056, 14, 14]))
start = time.time()
output = m(x410, x406)
end = time.time()
print(end-start)
