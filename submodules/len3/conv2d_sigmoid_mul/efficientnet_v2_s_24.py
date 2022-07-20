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
        self.conv2d142 = Conv2d(64, 1536, kernel_size=(1, 1), stride=(1, 1))
        self.sigmoid24 = Sigmoid()

    def forward(self, x452, x449):
        x453=self.conv2d142(x452)
        x454=self.sigmoid24(x453)
        x455=operator.mul(x454, x449)
        return x455

m = M().eval()
x452 = torch.randn(torch.Size([1, 64, 1, 1]))
x449 = torch.randn(torch.Size([1, 1536, 7, 7]))
start = time.time()
output = m(x452, x449)
end = time.time()
print(end-start)
