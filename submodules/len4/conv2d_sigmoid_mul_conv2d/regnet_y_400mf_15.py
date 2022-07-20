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
        self.conv2d83 = Conv2d(110, 440, kernel_size=(1, 1), stride=(1, 1))
        self.sigmoid15 = Sigmoid()
        self.conv2d84 = Conv2d(440, 440, kernel_size=(1, 1), stride=(1, 1), bias=False)

    def forward(self, x260, x257):
        x261=self.conv2d83(x260)
        x262=self.sigmoid15(x261)
        x263=operator.mul(x262, x257)
        x264=self.conv2d84(x263)
        return x264

m = M().eval()
x260 = torch.randn(torch.Size([1, 110, 1, 1]))
x257 = torch.randn(torch.Size([1, 440, 7, 7]))
start = time.time()
output = m(x260, x257)
end = time.time()
print(end-start)