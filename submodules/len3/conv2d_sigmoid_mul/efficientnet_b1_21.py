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
        self.conv2d107 = Conv2d(48, 1152, kernel_size=(1, 1), stride=(1, 1))
        self.sigmoid21 = Sigmoid()

    def forward(self, x330, x327):
        x331=self.conv2d107(x330)
        x332=self.sigmoid21(x331)
        x333=operator.mul(x332, x327)
        return x333

m = M().eval()
x330 = torch.randn(torch.Size([1, 48, 1, 1]))
x327 = torch.randn(torch.Size([1, 1152, 7, 7]))
start = time.time()
output = m(x330, x327)
end = time.time()
print(end-start)
