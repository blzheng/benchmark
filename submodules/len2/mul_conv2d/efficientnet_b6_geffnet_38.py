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
        self.conv2d192 = Conv2d(2064, 344, kernel_size=(1, 1), stride=(1, 1), bias=False)

    def forward(self, x567, x572):
        x573=operator.mul(x567, x572)
        x574=self.conv2d192(x573)
        return x574

m = M().eval()
x567 = torch.randn(torch.Size([1, 2064, 7, 7]))
x572 = torch.randn(torch.Size([1, 2064, 1, 1]))
start = time.time()
output = m(x567, x572)
end = time.time()
print(end-start)
