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
        self.sigmoid3 = Sigmoid()
        self.conv2d38 = Conv2d(512, 128, kernel_size=(1, 1), stride=(1, 1), bias=False)

    def forward(self, x121, x117):
        x122=self.sigmoid3(x121)
        x123=operator.mul(x122, x117)
        x124=self.conv2d38(x123)
        return x124

m = M().eval()
x121 = torch.randn(torch.Size([1, 512, 1, 1]))
x117 = torch.randn(torch.Size([1, 512, 14, 14]))
start = time.time()
output = m(x121, x117)
end = time.time()
print(end-start)
