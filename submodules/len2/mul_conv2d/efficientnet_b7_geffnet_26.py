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
        self.conv2d131 = Conv2d(960, 160, kernel_size=(1, 1), stride=(1, 1), bias=False)

    def forward(self, x387, x392):
        x393=operator.mul(x387, x392)
        x394=self.conv2d131(x393)
        return x394

m = M().eval()
x387 = torch.randn(torch.Size([1, 960, 14, 14]))
x392 = torch.randn(torch.Size([1, 960, 1, 1]))
start = time.time()
output = m(x387, x392)
end = time.time()
print(end-start)
