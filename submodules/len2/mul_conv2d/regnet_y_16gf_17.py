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
        self.conv2d94 = Conv2d(3024, 3024, kernel_size=(1, 1), stride=(1, 1), bias=False)

    def forward(self, x294, x289):
        x295=operator.mul(x294, x289)
        x296=self.conv2d94(x295)
        return x296

m = M().eval()
x294 = torch.randn(torch.Size([1, 3024, 1, 1]))
x289 = torch.randn(torch.Size([1, 3024, 7, 7]))
start = time.time()
output = m(x294, x289)
end = time.time()
print(end-start)
