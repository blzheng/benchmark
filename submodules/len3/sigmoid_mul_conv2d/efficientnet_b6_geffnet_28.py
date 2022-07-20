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
        self.conv2d142 = Conv2d(1200, 200, kernel_size=(1, 1), stride=(1, 1), bias=False)

    def forward(self, x422, x418):
        x423=x422.sigmoid()
        x424=operator.mul(x418, x423)
        x425=self.conv2d142(x424)
        return x425

m = M().eval()
x422 = torch.randn(torch.Size([1, 1200, 1, 1]))
x418 = torch.randn(torch.Size([1, 1200, 14, 14]))
start = time.time()
output = m(x422, x418)
end = time.time()
print(end-start)
