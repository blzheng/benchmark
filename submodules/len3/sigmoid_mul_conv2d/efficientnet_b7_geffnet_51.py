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
        self.conv2d256 = Conv2d(2304, 640, kernel_size=(1, 1), stride=(1, 1), bias=False)

    def forward(self, x764, x760):
        x765=x764.sigmoid()
        x766=operator.mul(x760, x765)
        x767=self.conv2d256(x766)
        return x767

m = M().eval()
x764 = torch.randn(torch.Size([1, 2304, 1, 1]))
x760 = torch.randn(torch.Size([1, 2304, 7, 7]))
start = time.time()
output = m(x764, x760)
end = time.time()
print(end-start)
