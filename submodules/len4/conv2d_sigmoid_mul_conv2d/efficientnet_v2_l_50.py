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
        self.conv2d286 = Conv2d(96, 2304, kernel_size=(1, 1), stride=(1, 1))
        self.sigmoid50 = Sigmoid()
        self.conv2d287 = Conv2d(2304, 384, kernel_size=(1, 1), stride=(1, 1), bias=False)

    def forward(self, x920, x917):
        x921=self.conv2d286(x920)
        x922=self.sigmoid50(x921)
        x923=operator.mul(x922, x917)
        x924=self.conv2d287(x923)
        return x924

m = M().eval()
x920 = torch.randn(torch.Size([1, 96, 1, 1]))
x917 = torch.randn(torch.Size([1, 2304, 7, 7]))
start = time.time()
output = m(x920, x917)
end = time.time()
print(end-start)
